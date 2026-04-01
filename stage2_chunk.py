#!/usr/bin/env python3
"""
One parquet + matching archives at a time (disk-light), aligned with deepfusion WebDataset upload style:

  1) snapshot_download: this parquet + your tar / tar.split globs (upload uses ``gcloud storage cp -n``; each sample is ``.jpg`` / ``.jpg2`` / ``.txt``, same idea as deepfusion MagicBrush).
  2) merge ``*.tar.split.NNN``, extract; unpack whole ``.tar`` blobs if present.
  3) audit paths vs files on disk (strict single in / single out); drops bad/missing rows by not uploading them.
  4) build + valid upload only for rows that resolve, decode, and JPEG-encode; fail if nothing passes (like MagicBrush “no samples”).
  5) optional ``--delete-chunk-after``: remove downloaded parquet + extracted trees to free disk (audit JSON copied next to the build report).

Hybrid shards: paths often use ``results_compose_partN`` (and ``…_part6_fix``) while Hub ships ``results_hybrid`` tars. For ``Parquet/hybrid_part*.parquet``, compose→hybrid subst is **auto-enabled** unless ``--no-auto-hybrid-compose``; you can still force it with ``--fix-hybrid-compose-paths``. Always set ``--allow-pattern`` to the Hub’s ``results_hybrid*`` archive globs.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from huggingface_hub import snapshot_download

from imgedit_lib import (
    HYBRID_COMPOSE_SUBST,
    log_paths_disk_usage,
    warn_bucket_zone,
    write_json,
)
from stage2_build_webdataset import (
    DEFAULT_BUCKET,
    DEFAULT_REPO,
    audit_parquet_resolution,
    parse_substs,
    run_build,
)

logger = logging.getLogger(__name__)

TAR_SPLIT_RE = re.compile(r"^(.+\.tar)\.split\.(\d+)$")


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
        root.addHandler(h)


def normalize_repo_path(p: str) -> str:
    p = p.strip().lstrip("./")
    return p.lstrip("/")


def merge_and_extract_tar_splits(chunk_dir: Path) -> int:
    """
    Find files named <name>.tar.split.NNN, merge pieces in numeric order, extract under the
    same directory, then remove split files. Returns the number of merged archives processed.
    """
    groups: Dict[Tuple[Path, str], List[Tuple[int, Path]]] = {}
    for path in chunk_dir.rglob("*"):
        if not path.is_file():
            continue
        m = TAR_SPLIT_RE.match(path.name)
        if not m:
            continue
        base_tar_name = m.group(1)
        idx = int(m.group(2))
        key = (path.parent, base_tar_name)
        groups.setdefault(key, []).append((idx, path))

    n_done = 0
    for (parent, base_tar_name) in sorted(groups.keys()):
        parts = groups[(parent, base_tar_name)]
        parts.sort(key=lambda x: x[0])
        sorted_idx = [i for i, _ in parts]
        span = list(range(sorted_idx[0], sorted_idx[-1] + 1))
        if sorted_idx != span:
            raise ValueError(
                f"Incomplete tar splits for {parent / base_tar_name}: "
                f"have indices {sorted_idx}, expected contiguous {span[0]}..{span[-1]}"
            )
        merged = b"".join(p.read_bytes() for _, p in parts)
        bio = io.BytesIO(merged)
        with tarfile.open(fileobj=bio, mode="r:*") as tf:
            tf.extractall(path=parent)
        n_done += 1
        logger.info("Merged + extracted %s (%d parts) under %s", base_tar_name, len(parts), parent)
        for _, pf in parts:
            pf.unlink(missing_ok=True)
    return n_done


def extract_standalone_tars(chunk_dir: Path) -> int:
    """
    Extract whole `.tar` files under chunk_dir (not `*.tar.split.NNN` pieces).
    Hugging Face may ship monolithic archives alongside split parts; MagicBrush-style flow
    still expects a flat tree of image files for parquet path resolution.
    """
    n = 0
    for path in sorted(chunk_dir.rglob("*.tar")):
        if not path.is_file():
            continue
        if TAR_SPLIT_RE.match(path.name):
            continue
        try:
            with tarfile.open(path, "r:*") as tf:
                tf.extractall(path=path.parent)
        except tarfile.TarError as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue
        n += 1
        logger.info("Extracted standalone %s → %s", path.name, path.parent)
        path.unlink(missing_ok=True)
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ImgEdit: one parquet + tars — download, extract splits, audit, build, delete chunk."
    )
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument(
        "--chunk-dir",
        required=True,
        help="HF local_dir and dataset root for this slice.",
    )
    p.add_argument(
        "--parquet",
        required=True,
        help="Repo-relative parquet path, e.g. Parquet/remove_part0.parquet (always downloaded).",
    )
    p.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Extra snapshot_download allow_patterns (repeat), e.g. Singleturn/results_*.tar.split.*",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only snapshot_download; exit before extract / audit / build.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing chunk-dir contents.",
    )
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip all automatic extract: no *.tar.split.* merge and no standalone .tar unpack.",
    )
    p.add_argument(
        "--delete-chunk-after",
        action="store_true",
        help="After a successful build, remove --chunk-dir (verify path before enabling).",
    )
    p.add_argument(
        "--shard-prefix",
        default="",
        help="Shard filename prefix (default: parquet stem, e.g. remove_part0).",
    )
    p.add_argument("--skip-audit", action="store_true", help="Skip parquet↔disk path audit (not recommended).")
    p.add_argument(
        "--min-resolve-rate",
        type=float,
        default=1.0,
        help="Minimum fraction of strict pairs whose input+output paths exist (default: 1.0).",
    )
    p.add_argument(
        "--allow-zero-strict-pairs",
        action="store_true",
        help="Allow parquets that emit no strict pairs (e.g. multi-input rows only); build may write nothing.",
    )
    p.add_argument("--work-dir", default="/dev/shm/imgedit_wds", help="Temp dir for shard .tar files before upload.")
    p.add_argument(
        "--bucket",
        default="",
        help=(
            "gs://…/prefix; uploads with gcloud storage cp -n when set (typical "
            f"{DEFAULT_BUCKET} — set zone to your VM). Empty = keep shards under --work-dir only."
        ),
    )
    p.add_argument(
        "--expected-zone",
        default="",
        help="Warn if this token is missing from --bucket (e.g. us-central1 when using regional buckets).",
    )
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-side", type=int, default=2048)
    p.add_argument("--sanity-decode", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--subst", action="append", default=[], metavar="OLD:NEW")
    p.add_argument(
        "--fix-hybrid-compose-paths",
        action="store_true",
        help=(
            "Apply compose→hybrid path subst (results_compose→results_hybrid, first match per path). "
            "Also on by default for parquet stem hybrid_part* (see --no-auto-hybrid-compose)."
        ),
    )
    p.add_argument(
        "--no-auto-hybrid-compose",
        action="store_true",
        help="Disable automatic compose→hybrid subst for Parquet/hybrid_part*.parquet (stage1 logs warn on results_compose_* roots).",
    )
    p.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Do not fail when 0 samples pass JPEG validation (default: fail; unsafe for production chunks).",
    )
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--report-path", default="", help="Default: <work-dir>/chunk_build_report.json")
    return p.parse_args()


def _merge_subst_rules(
    user_rules: Sequence[Tuple[str, str]], *, hybrid_fix: bool
) -> List[Tuple[str, str]]:
    out = list(user_rules)
    if hybrid_fix:
        out.insert(0, HYBRID_COMPOSE_SUBST)
    return out


def _hybrid_compose_fix_mode(
    *, pq_stem: str, explicit: bool, no_auto: bool
) -> Tuple[bool, str]:
    """Returns (apply_fix, reason) for logging."""
    if explicit:
        return True, "--fix-hybrid-compose-paths"
    if no_auto:
        return False, ""
    if pq_stem.startswith("hybrid_part"):
        return True, f"auto(stem {pq_stem!r} matches hybrid_part*)"
    return False, ""


def main() -> None:
    setup_logging()
    args = parse_args()
    warn_bucket_zone(args.bucket, args.expected_zone)
    chunk = Path(args.chunk_dir).resolve()
    work = Path(args.work_dir)
    pq_rel = normalize_repo_path(args.parquet)
    pq_basename = Path(pq_rel).name
    pq_stem = Path(pq_rel).stem
    parquets_filter = {pq_basename}
    shard_prefix = (args.shard_prefix or pq_stem).strip()
    hybrid_fix, hybrid_reason = _hybrid_compose_fix_mode(
        pq_stem=pq_stem,
        explicit=bool(args.fix_hybrid_compose_paths),
        no_auto=bool(args.no_auto_hybrid_compose),
    )
    subst_rules = _merge_subst_rules(parse_substs(args.subst), hybrid_fix=hybrid_fix)
    if hybrid_fix:
        logger.info(
            "Compose→hybrid path subst [%s]: %r→%r (first match per path). "
            "Download matching results_hybrid* archives via --allow-pattern.",
            hybrid_reason,
            HYBRID_COMPOSE_SUBST[0],
            HYBRID_COMPOSE_SUBST[1],
        )

    if args.delete_chunk_after and not (args.bucket or "").strip():
        logger.warning(
            "--delete-chunk-after with no --bucket: raw slice will be removed but nothing was uploaded to GCS from this run."
        )

    if args.skip_download and args.download_only:
        raise SystemExit("Choose at most one of --skip-download / --download-only.")

    if not args.skip_download:
        if not args.allow_pattern:
            raise SystemExit(
                "Pass at least one --allow-pattern for image archives "
                "(the parquet is added automatically), or use --skip-download."
            )
        chunk.mkdir(parents=True, exist_ok=True)
        log_paths_disk_usage("pre_download", [chunk, work])
        patterns: List[str] = [pq_rel] + list(args.allow_pattern)
        logger.info("snapshot_download %s → %s patterns=%s", args.repo, chunk, patterns)
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(chunk),
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
        )
        logger.info("Download finished.")

    if args.download_only:
        logger.info("Download-only stop. Re-run without --download-only to extract, audit, build, and optionally delete.")
        return

    if not args.no_extract:
        n_split = merge_and_extract_tar_splits(chunk)
        n_tar = extract_standalone_tars(chunk)
        logger.info(
            "Extract done: %d split merge(s), %d standalone .tar(s).",
            n_split,
            n_tar,
        )
        log_paths_disk_usage("post_extract", [chunk, work])

    pq_local = chunk / pq_rel
    if not pq_local.is_file():
        raise SystemExit(f"Missing parquet at {pq_local} (check --parquet and --chunk-dir).")

    if not args.skip_audit:
        audit = audit_parquet_resolution(
            pq_local,
            chunk,
            subst_rules,
            batch_size=args.batch_size,
        )
        audit_path = chunk / f"audit_{pq_stem}.json"
        audit["subst_rules_applied"] = [[a, b] for a, b in subst_rules]
        write_json(audit_path, audit)
        logger.info("Wrote audit %s", audit_path)

        if audit.get("error") == "unknown_schema":
            raise SystemExit(f"Audit failed: unknown schema for {pq_basename}")

        strict = audit.get("strict_pairs", 0)
        if strict == 0 and not args.allow_zero_strict_pairs:
            raise SystemExit(
                f"Audit: 0 strict pairs in {pq_basename} (multi-input-only or empty?). "
                f"Use --allow-zero-strict-pairs to force build anyway."
            )

        if strict > 0:
            rate = audit.get("resolve_rate")
            if rate is None or rate < args.min_resolve_rate:
                logger.error(
                    "Audit: resolve_rate=%s (need >= %s). broken_pairs=%s examples=%s",
                    rate,
                    args.min_resolve_rate,
                    audit.get("broken_pairs"),
                    audit.get("examples_broken"),
                )
                raise SystemExit(
                    f"Audit failed: parquet paths vs on-disk files (resolve_rate {rate}, "
                    f"min {args.min_resolve_rate}). Fix downloads/extract or set --min-resolve-rate."
                )

    report_path = Path(args.report_path) if args.report_path else work / "chunk_build_report.json"
    log_paths_disk_usage("pre_build", [chunk, work])

    try:
        summary = run_build(
            dataset_root=chunk,
            work_dir=work,
            bucket=args.bucket,
            subst_rules=subst_rules,
            samples_per_shard=args.samples_per_shard,
            batch_size=args.batch_size,
            jpeg_quality=args.jpeg_quality,
            max_side=args.max_side,
            sanity_decode=args.sanity_decode,
            skip_existing=args.skip_existing,
            max_shards=args.max_shards,
            report_path=report_path,
            shard_prefix=shard_prefix,
            parquets_filter=parquets_filter,
        )
    except Exception:
        logger.exception("Build failed; chunk-dir left at %s", chunk)
        raise

    total_kept = int(summary.get("total_kept", 0))
    if total_kept == 0 and not args.allow_empty_output:
        raise SystemExit(
            "No samples passed validation (paths + decode + JPEG). "
            "Chunk dir kept for debugging. Fix data/patterns/subst or pass --allow-empty-output to override."
        )

    if args.delete_chunk_after:
        preserved = chunk / f"audit_{pq_stem}.json"
        if preserved.is_file():
            dest = report_path.parent / preserved.name
            shutil.copy2(preserved, dest)
            logger.info("Preserved audit → %s", dest)
        logger.info("Removing chunk dir %s", chunk)
        shutil.rmtree(chunk)


if __name__ == "__main__":
    main()
