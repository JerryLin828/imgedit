#!/usr/bin/env python3
"""
One parquet + matching archives at a time (disk-light), aligned with deepfusion WebDataset upload style:

  1) snapshot_download: this parquet + your tar / tar.split globs (upload uses ``gcloud storage cp -n``; each sample is ``.jpg`` / ``.jpg2`` / ``.txt``, same idea as deepfusion MagicBrush).
  2) merge ``*.tar.split.NNN``, extract; unpack whole ``.tar`` blobs if present.
  3) audit paths vs files on disk (strict single in / single out); drops bad/missing rows by not uploading them.
  4) build + valid upload only for rows that resolve, decode, and JPEG-encode; fail if nothing passes (like MagicBrush “no samples”).
  5) optional ``--delete-chunk-after``: remove downloaded parquet + extracted trees to free disk (audit JSON copied next to the build report).
  6) optional ``--delete-chunk-on-failure``: on audit failure or zero kept samples, also remove chunk dir (same audit copy as success).

Exit codes: 0 = success; 2 = audit below ``--min-resolve-rate``, zero strict pairs, unknown schema, or zero kept samples; 1 = other failures (CPython argparse may also use exit 2 for usage errors).

Hybrid shards: paths often use ``results_compose_partN`` (and ``…_part6_fix``) while Hub ships ``results_hybrid`` tars. For ``Parquet/hybrid_part*.parquet``, compose→hybrid subst is **auto-enabled** unless ``--no-auto-hybrid-compose``; you can still force it with ``--fix-hybrid-compose-paths``. Always set ``--allow-pattern`` to the Hub’s ``results_hybrid*`` archive globs.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tarfile
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple

import requests
from huggingface_hub import hf_hub_url, list_repo_files, snapshot_download
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# Distinct for shell orchestration (e.g. run_all_chunks.sh). Note: argparse also uses code 2 for usage errors.
EXIT_GENERIC_FAILURE = 1
EXIT_AUDIT_OR_EMPTY_SKIP = 2

TAR_SPLIT_RE = re.compile(r"^(.+\.tar)\.split\.(\d+)$")

# `tarfile` streaming reads; keep each read() bounded so we never load a whole split or the full merged tar.
_MERGE_READ_BUFSIZE = 4 * 1024 * 1024

# Direct HTTP download defaults (mirrors deepfusion/dataset/upload_mario10m.py style).
_DL_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MiB per iter_content chunk
_DL_TIMEOUT = 300.0
_DL_RETRIES = 5
_DL_BACKOFF = 1.0


def _make_download_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_DL_RETRIES,
        backoff_factor=_DL_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _direct_download_file(
    repo_id: str,
    repo_path: str,
    local_dir: Path,
    token: Optional[str],
    session: requests.Session,
    chunk_size: int = _DL_CHUNK_SIZE,
    timeout: float = _DL_TIMEOUT,
    max_attempts: int = 3,
) -> Path:
    """
    Stream one file from HF via resolved CDN URL (bypasses XET/snapshot_download overhead).

    Writes to ``.part`` temp, atomic-renames on success. Skips if dest already exists with
    matching ``Content-Length``.
    """
    dest = local_dir / repo_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(repo_id=repo_id, filename=repo_path, repo_type="dataset")
    headers: Dict[str, str] = {}
    if token:
        headers["authorization"] = f"Bearer {token}"

    expected: Optional[int] = None
    try:
        head = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        head.raise_for_status()
        cl = head.headers.get("Content-Length")
        if cl is not None:
            expected = int(cl)
    except Exception:
        pass

    if dest.is_file():
        if expected is None or dest.stat().st_size == expected:
            logger.info("Skip (already complete) %s", repo_path)
            return dest
        logger.info("Incomplete %s on disk (%d vs expected %d); redownloading.",
                     repo_path, dest.stat().st_size, expected)
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        if part.exists():
            part.unlink()
        written = 0
        t0 = time.monotonic()
        try:
            with session.get(url, stream=True, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fp:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fp.write(chunk)
                            written += len(chunk)
        except Exception as exc:
            if part.exists():
                part.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise
            logger.warning("Download error %s (attempt %d/%d): %s; retrying…",
                           repo_path, attempt, max_attempts, exc)
            time.sleep(_DL_BACKOFF * attempt)
            continue

        if expected is not None and written != expected:
            if part.exists():
                part.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise IOError(
                    f"Size mismatch for {repo_path}: expected {expected}, got {written}"
                )
            logger.warning("Size mismatch %s (%d vs %d, attempt %d/%d); retrying…",
                           repo_path, written, expected, attempt, max_attempts)
            continue

        shutil.move(str(part), str(dest))
        elapsed = time.monotonic() - t0
        speed = (written / (1024**2)) / max(elapsed, 0.01)
        logger.info("Downloaded %s (%.2f MiB, %.1f MiB/s)", repo_path, written / (1024**2), speed)
        return dest

    raise RuntimeError(f"Failed to download {repo_path} after {max_attempts} attempts")


class _TarSplitConcatReader:
    """
    File-like object that yields split parts in order for ``tarfile.open(..., mode='r|*')``.
    Avoids BytesIO (~full archive RAM) and avoids writing a second full ``.tar`` to disk.

    After each part is fully read, it is **unlinked** immediately so peak disk can drop from
    ~(T + E) toward ~(max(T, E)) in the best case (T = total split bytes, E = extracted payload),
    depending on how tar members interleave with split boundaries.
    """

    def __init__(self, paths: List[Path]) -> None:
        self._paths = paths
        self._idx = 0
        self._fp: Optional[BinaryIO] = None

    def _unlink_consumed_split(self) -> None:
        """Remove the split file we just finished reading (index _idx - 1)."""
        prev_i = self._idx - 1
        if prev_i < 0:
            return
        done = self._paths[prev_i]
        try:
            done.unlink(missing_ok=True)
            logger.debug("Removed consumed split %s", done.name)
        except OSError as exc:
            logger.warning("Could not remove consumed split %s: %s", done, exc)

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks: List[bytes] = []
        want = size
        while want < 0 or want > 0:
            if self._fp is None:
                if self._idx >= len(self._paths):
                    break
                self._fp = open(self._paths[self._idx], "rb")
                self._idx += 1
            to_read = _MERGE_READ_BUFSIZE if want < 0 else min(_MERGE_READ_BUFSIZE, want)
            piece = self._fp.read(to_read)
            if not piece:
                self._fp.close()
                self._fp = None
                self._unlink_consumed_split()
                continue
            chunks.append(piece)
            if want >= 0:
                want -= len(piece)
        return b"".join(chunks)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


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


def _preserve_audit_and_rmtree_chunk(chunk: Path, report_path: Path, pq_stem: str) -> None:
    """Copy audit JSON next to build report, then remove chunk dir (same as --delete-chunk-after)."""
    preserved = chunk / f"audit_{pq_stem}.json"
    if preserved.is_file():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        dest = report_path.parent / preserved.name
        shutil.copy2(preserved, dest)
        logger.info("Preserved audit → %s", dest)
    if chunk.is_dir():
        logger.info("Removing chunk dir %s", chunk)
        shutil.rmtree(chunk)


def _audit_skip_exit(
    message: str,
    *,
    chunk: Path,
    report_path: Path,
    pq_stem: str,
    delete_chunk_on_failure: bool,
) -> None:
    """Audit / zero-output skip: log, optionally delete chunk dir, exit with EXIT_AUDIT_OR_EMPTY_SKIP."""
    logger.error(message)
    if delete_chunk_on_failure:
        _preserve_audit_and_rmtree_chunk(chunk, report_path, pq_stem)
        logger.info("Chunk dir removed (--delete-chunk-on-failure).")
    sys.exit(EXIT_AUDIT_OR_EMPTY_SKIP)


def merge_and_extract_tar_splits(chunk_dir: Path) -> int:
    """
    Find files named <name>.tar.split.NNN, merge pieces in numeric order, extract under the
    same directory, then remove split files. Returns the number of merged archives processed.

    Merging uses ``tarfile`` streaming mode over a concatenated reader so RAM stays bounded
    (no full-archive BytesIO / ``b"".join``). Each ``.split.NNN`` is deleted as soon as it is
    fully read, which reduces **peak** disk from ~**(T + E)** toward often **~(max(T, E))** when
    layout allows (never below **T** while the first split is still downloading / present).
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
        ordered_paths = [p for _, p in parts]
        total_b = sum(p.stat().st_size for p in ordered_paths)
        cat = _TarSplitConcatReader(ordered_paths)
        try:
            with tarfile.open(fileobj=cat, mode="r|*") as tf:
                tf.extractall(path=parent)
        finally:
            cat.close()
            for p in ordered_paths:
                p.unlink(missing_ok=True)
        n_done += 1
        logger.info(
            "Merged + extracted %s (%d parts, ~%.2f GiB) under %s (streaming; splits trimmed as read)",
            base_tar_name,
            len(parts),
            total_b / (1024**3),
            parent,
        )
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
        "--direct-download",
        action="store_true",
        help=(
            "Use direct HTTP streaming (requests + hf_hub_url) instead of snapshot_download for archive files. "
            "Much faster for large splits — bypasses XET/HF cache overhead. "
            "Reads HF_TOKEN from environment for auth."
        ),
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download; exit before extract / audit / build.",
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
        "--delete-chunk-on-failure",
        dest="delete_chunk_on_failure",
        action="store_true",
        help=(
            "On audit failure (below min-resolve-rate, zero strict pairs, unknown schema) or "
            "zero kept samples, copy audit next to report and remove --chunk-dir before exit "
            f"(exit {EXIT_AUDIT_OR_EMPTY_SKIP})."
        ),
    )
    p.add_argument(
        "--shard-prefix",
        default="",
        help=(
            "Prefix for shard basenames (empty = MagicBrush-style shard-00000.tar; "
            "set explicitly, e.g. remove_part0, when using a flat GCS prefix without per-chunk subdirs)."
        ),
    )
    p.add_argument(
        "--image-subdir",
        default="",
        help=(
            "Prepend this subdirectory to bare image filenames (no directory component) before path resolution. "
            "E.g. Singleturn/part1 for action_part* chunks."
        ),
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
    report_path = Path(args.report_path) if args.report_path else work / "chunk_build_report.json"
    parquets_filter = {pq_basename}
    shard_prefix = args.shard_prefix.strip()
    image_subdir = (args.image_subdir or "").strip()
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
        logger.error("Choose at most one of --skip-download / --download-only.")
        sys.exit(EXIT_GENERIC_FAILURE)

    if not args.skip_download:
        if not args.allow_pattern:
            logger.error(
                "Pass at least one --allow-pattern for image archives "
                "(the parquet is added automatically), or use --skip-download."
            )
            sys.exit(EXIT_GENERIC_FAILURE)
        chunk.mkdir(parents=True, exist_ok=True)
        log_paths_disk_usage("pre_download", [chunk, work])

        logger.info("Downloading parquet %s", pq_rel)
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(chunk),
            local_dir_use_symlinks=False,
            allow_patterns=[pq_rel],
        )

        all_files = sorted(list_repo_files(args.repo, repo_type="dataset"))

        if args.direct_download:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not set; --direct-download may fail for gated datasets.")
            session = _make_download_session()
            for pattern in args.allow_pattern:
                matched = sorted(f for f in all_files if fnmatch(f, pattern))
                logger.info("Pattern %s → %d file(s) [direct HTTP]", pattern, len(matched))
                for rel in matched:
                    _direct_download_file(
                        repo_id=args.repo,
                        repo_path=rel,
                        local_dir=chunk,
                        token=token,
                        session=session,
                    )
        else:
            for pattern in args.allow_pattern:
                matched = sorted(f for f in all_files if fnmatch(f, pattern))
                logger.info("Pattern %s → %d file(s)", pattern, len(matched))
                for rel in matched:
                    logger.info("Downloading %s", rel)
                    snapshot_download(
                        repo_id=args.repo,
                        repo_type="dataset",
                        local_dir=str(chunk),
                        local_dir_use_symlinks=False,
                        allow_patterns=[rel],
                    )

        logger.info("Download finished (%d pattern(s), repo file list size %d).",
                     len(args.allow_pattern), len(all_files))

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
        logger.error("Missing parquet at %s (check --parquet and --chunk-dir).", pq_local)
        sys.exit(EXIT_GENERIC_FAILURE)

    if not args.skip_audit:
        audit = audit_parquet_resolution(
            pq_local,
            chunk,
            subst_rules,
            batch_size=args.batch_size,
            image_subdir=image_subdir,
        )
        audit_path = chunk / f"audit_{pq_stem}.json"
        audit["subst_rules_applied"] = [[a, b] for a, b in subst_rules]
        if image_subdir:
            audit["image_subdir"] = image_subdir
        write_json(audit_path, audit)
        logger.info("Wrote audit %s", audit_path)

        if audit.get("error") == "unknown_schema":
            _audit_skip_exit(
                f"Audit failed: unknown schema for {pq_basename}",
                chunk=chunk,
                report_path=report_path,
                pq_stem=pq_stem,
                delete_chunk_on_failure=args.delete_chunk_on_failure,
            )

        strict = audit.get("strict_pairs", 0)
        if strict == 0 and not args.allow_zero_strict_pairs:
            _audit_skip_exit(
                f"Audit: 0 strict pairs in {pq_basename} (multi-input-only or empty?). "
                f"Use --allow-zero-strict-pairs to force build anyway.",
                chunk=chunk,
                report_path=report_path,
                pq_stem=pq_stem,
                delete_chunk_on_failure=args.delete_chunk_on_failure,
            )

        if strict > 0:
            rate = audit.get("resolve_rate")
            if rate is None or rate < args.min_resolve_rate:
                _audit_skip_exit(
                    f"Audit failed: resolve_rate={rate} (need >= {args.min_resolve_rate}); "
                    f"broken_pairs={audit.get('broken_pairs')} examples={audit.get('examples_broken')}",
                    chunk=chunk,
                    report_path=report_path,
                    pq_stem=pq_stem,
                    delete_chunk_on_failure=args.delete_chunk_on_failure,
                )

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
            image_subdir=image_subdir,
        )
    except Exception:
        logger.exception("Build failed; chunk-dir left at %s", chunk)
        raise

    if not isinstance(summary, dict):
        logger.error("run_build returned unexpected value (expected dict): %r", summary)
        sys.exit(EXIT_GENERIC_FAILURE)

    total_kept = int(summary.get("total_kept", 0))
    if total_kept == 0 and not args.allow_empty_output:
        _audit_skip_exit(
            "No samples passed validation (paths + decode + JPEG). "
            "Fix data/patterns/subst or pass --allow-empty-output to override.",
            chunk=chunk,
            report_path=report_path,
            pq_stem=pq_stem,
            delete_chunk_on_failure=args.delete_chunk_on_failure,
        )

    if args.delete_chunk_after:
        _preserve_audit_and_rmtree_chunk(chunk, report_path, pq_stem)


if __name__ == "__main__":
    main()
