#!/usr/bin/env python3
"""
Stage 2 — Strict WebDataset shards + optional GCS upload.

Keeps a sample only when exactly one input path and one output path are declared,
both files exist, read successfully, and JPEG re-encode succeeds.
Optional --subst OLD:NEW only after you verified renames on disk.

Use --shard-prefix when processing chunks so GCS object names do not collide across runs.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from imgedit_lib import (
    add_tar_bytes,
    apply_image_subdir,
    detect_field_mapping,
    encode_jpeg,
    iter_imgedit_pairs_from_row,
    parquet_schema_columns,
    path_string_to_bytes,
    resolve_dataset_image_path,
    sanity_check_tar_triplet,
    upload_shard_noclobber,
    gcs_object_exists,
    warn_bucket_zone,
    write_json,
)

DEFAULT_REPO = "sysuyy/ImgEdit"
# Same layout convention as deepfusion/dataset/upload_magicbrush_webdataset.py (override per zone).
DEFAULT_BUCKET = "gs://kmh-gcp-us-central1/data/imgedit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: strict WebDataset build + GCS upload.")
    p.add_argument("--repo", default=DEFAULT_REPO, help="Only with --full-snapshot-download.")
    p.add_argument(
        "--dataset-root",
        default="",
        help="Root with Parquet/ and extracted images.",
    )
    p.add_argument(
        "--full-snapshot-download",
        action="store_true",
        help="Download full repo into --dataset-root (very large).",
    )
    p.add_argument("--work-dir", default="/dev/shm/imgedit_wds", help="Temporary shards directory.")
    p.add_argument(
        "--bucket",
        default="",
        help=f"gs://bucket/prefix — empty keeps .tar local only. Typical: {DEFAULT_BUCKET} (set zone to match VM).",
    )
    p.add_argument(
        "--expected-zone",
        default="",
        help="If set, warn when this token is not contained in --bucket (deepfusion-style safety check).",
    )
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-side", type=int, default=2048, help="0 = no resize.")
    p.add_argument("--sanity-decode", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--subst",
        action="append",
        default=[],
        metavar="OLD:NEW",
        help="Verified path substring replace (repeatable).",
    )
    p.add_argument("--max-shards", type=int, default=0, help="Stop after N shards (debug).")
    p.add_argument("--report-path", default="", help="Default: <work-dir>/build_report.json")
    p.add_argument(
        "--shard-prefix",
        default="",
        help="Prefix for shard file names on disk/GCS, e.g. remove_part0 → remove_part0-shard-00000.tar (avoid collisions between chunk runs).",
    )
    p.add_argument(
        "--parquets-only",
        default="",
        help="Comma-separated parquet basenames to process only, e.g. remove_part0.parquet,remove_part1.parquet",
    )
    p.add_argument(
        "--image-subdir",
        default="",
        help=(
            "Prepend this subdirectory to bare image filenames (no directory component) before path resolution. "
            "E.g. Singleturn/part1 for action_part* chunks."
        ),
    )
    return p.parse_args()


def parse_substs(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --subst (need OLD:NEW): {item}")
        old, new = item.split(":", 1)
        if not old:
            raise ValueError(f"Bad --subst: {item}")
        out.append((old, new))
    return out


def parse_parquets_only(s: str) -> Optional[Set[str]]:
    if not s.strip():
        return None
    names = {x.strip() for x in s.split(",") if x.strip()}
    return names if names else None


def _row_dicts_from_batch(batch) -> List[Dict[str, Any]]:
    d = batch.to_pydict()
    n = batch.num_rows
    keys = list(d.keys())
    return [{k: d[k][i] for k in keys} for i in range(n)]


def make_sample_key(parquet_stem: str, pair_index: int, prompt: str) -> str:
    h = hashlib.sha256(
        f"{parquet_stem}\0{pair_index}\0{prompt}".encode("utf-8", errors="replace")
    ).hexdigest()[:20]
    return f"{parquet_stem}_{pair_index:08d}_{h}"


def shard_basename(shard_idx: int, shard_prefix: str) -> str:
    body = f"shard-{shard_idx:05d}.tar"
    if shard_prefix.strip():
        return f"{shard_prefix.strip()}-{body}"
    return body


def audit_parquet_resolution(
    parquet_path: Path,
    dataset_root: Path,
    subst_rules: Sequence[Tuple[str, str]],
    *,
    batch_size: int = 2048,
    max_examples: int = 8,
    image_subdir: str = "",
) -> Dict[str, Any]:
    """
    Check that strict (single in / single out) paths in the parquet resolve to real files
    under dataset_root. Does not read image bytes; only path existence.
    """
    cols = parquet_schema_columns(parquet_path)
    mapping = detect_field_mapping(cols)
    mode, orig_c, edit_c, text_c = mapping[3], mapping[0], mapping[1], mapping[2]
    if mode == "unknown":
        return {"error": "unknown_schema", "parquet": parquet_path.name}

    stats: Dict[str, Any] = {
        "parquet": parquet_path.name,
        "rows": 0,
        "strict_pairs": 0,
        "both_paths_exist": 0,
        "broken_pairs": 0,
        "examples_broken": [],
    }

    def note_broken(inp: str, out: str, reason: str) -> None:
        if len(stats["examples_broken"]) >= max_examples:
            return
        stats["examples_broken"].append({"input": inp, "output": out, "reason": reason})

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        stats["rows"] += batch.num_rows
        for row in _row_dicts_from_batch(batch):
            for inp_rel, out_rel, _ in iter_imgedit_pairs_from_row(
                row, mode=mode, orig_col=orig_c or "", edit_col=edit_c or "", text_col=text_c
            ):
                stats["strict_pairs"] += 1
                inp_res = apply_image_subdir(inp_rel, image_subdir)
                out_res = apply_image_subdir(out_rel, image_subdir)
                in_ok = bool(
                    resolve_dataset_image_path(
                        inp_res,
                        dataset_root=dataset_root,
                        parquet_path=parquet_path,
                        subst_rules=subst_rules,
                    )
                )
                out_ok = bool(
                    resolve_dataset_image_path(
                        out_res,
                        dataset_root=dataset_root,
                        parquet_path=parquet_path,
                        subst_rules=subst_rules,
                    )
                )
                if in_ok and out_ok:
                    stats["both_paths_exist"] += 1
                else:
                    stats["broken_pairs"] += 1
                    if not in_ok and not out_ok:
                        note_broken(inp_res, out_res, "missing_both")
                    elif not in_ok:
                        note_broken(inp_res, out_res, "missing_input")
                    else:
                        note_broken(inp_res, out_res, "missing_output")

    sp = stats["strict_pairs"]
    if sp:
        stats["resolve_rate"] = stats["both_paths_exist"] / sp
    else:
        stats["resolve_rate"] = None
    return stats


def run_build(
    *,
    dataset_root: Path,
    work_dir: Path,
    bucket: str,
    subst_rules: Sequence[Tuple[str, str]],
    samples_per_shard: int,
    batch_size: int,
    jpeg_quality: int,
    max_side: Optional[int],
    sanity_decode: int,
    skip_existing: bool,
    max_shards: int,
    report_path: Path,
    shard_prefix: str = "",
    parquets_filter: Optional[Set[str]] = None,
    image_subdir: str = "",
) -> Dict[str, Any]:
    """Core build loop; returns summary dict."""
    work_dir.mkdir(parents=True, exist_ok=True)
    parquets = sorted(dataset_root.glob("Parquet/*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No Parquet/*.parquet under {dataset_root}")
    if parquets_filter is not None:
        parquets = [p for p in parquets if p.name in parquets_filter]
        if not parquets:
            raise FileNotFoundError(
                f"No parquets matched filter {parquets_filter!r} under {dataset_root / 'Parquet'}"
            )

    max_side_eff = None if max_side is not None and max_side <= 0 else max_side
    bucket_eff = (bucket or "").strip().rstrip("/")

    buffer: List[Tuple[str, bytes, bytes, bytes]] = []
    shard_idx = 0
    total_kept = 0
    per_file_stats: List[Dict[str, Any]] = []
    pair_index_global = 0

    def flush_shard() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        n = len(buffer)
        tar_name = shard_basename(shard_idx, shard_prefix)
        tar_path = work_dir / tar_name
        gcs_uri = f"{bucket_eff}/{tar_name}" if bucket_eff else ""

        if skip_existing and gcs_uri and gcs_object_exists(gcs_uri):
            print(f"  skip (exists) {gcs_uri}")
            buffer.clear()
            shard_idx += 1
            return

        # Recreate if missing (shared VM: tmpfs cleaner, stray rm -rf, or race with another job).
        work_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "w") as tf:
            for key, j1, j2, txt in buffer:
                add_tar_bytes(tf, f"{key}.jpg", j1)
                add_tar_bytes(tf, f"{key}.jpg2", j2)
                add_tar_bytes(tf, f"{key}.txt", txt)

        sanity_check_tar_triplet(str(tar_path), n, decode_samples=sanity_decode)
        if bucket_eff:
            upload_shard_noclobber(str(tar_path), bucket_eff + "/")
            tar_path.unlink(missing_ok=True)
        buffer.clear()
        shard_idx += 1
        print(f"  wrote {tar_name} ({n} samples)")

    for pq_path in parquets:
        stem = pq_path.stem
        cols = parquet_schema_columns(pq_path)
        mapping = detect_field_mapping(cols)
        mode, orig_c, edit_c, text_c = mapping[3], mapping[0], mapping[1], mapping[2]

        st: Dict[str, Any] = {
            "parquet": pq_path.name,
            "rows": 0,
            "pairs_emitted": 0,
            "kept": 0,
            "missing_file": 0,
            "bad_image": 0,
        }

        if mode == "unknown":
            st["note"] = "skipped_unknown_schema"
            per_file_stats.append(st)
            print(f"{stem}: skip (unknown schema)")
            continue

        print(f"{stem}: processing…")
        pf = pq.ParquetFile(pq_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            st["rows"] += batch.num_rows
            for row in _row_dicts_from_batch(batch):
                for inp_rel, out_rel, prompt in iter_imgedit_pairs_from_row(
                    row, mode=mode, orig_col=orig_c or "", edit_col=edit_c or "", text_col=text_c
                ):
                    st["pairs_emitted"] += 1
                    inp_res = apply_image_subdir(inp_rel, image_subdir)
                    out_res = apply_image_subdir(out_rel, image_subdir)
                    if not (
                        resolve_dataset_image_path(
                            inp_res,
                            dataset_root=dataset_root,
                            parquet_path=pq_path,
                            subst_rules=subst_rules,
                        )
                        and resolve_dataset_image_path(
                            out_res,
                            dataset_root=dataset_root,
                            parquet_path=pq_path,
                            subst_rules=subst_rules,
                        )
                    ):
                        st["missing_file"] += 1
                        continue

                    raw_in = path_string_to_bytes(
                        inp_res,
                        dataset_root=dataset_root,
                        parquet_path=pq_path,
                        subst_rules=subst_rules,
                    )
                    raw_out = path_string_to_bytes(
                        out_res,
                        dataset_root=dataset_root,
                        parquet_path=pq_path,
                        subst_rules=subst_rules,
                    )
                    if not raw_in or not raw_out:
                        st["missing_file"] += 1
                        continue

                    jpg_in = encode_jpeg(raw_in, max_side=max_side_eff, jpeg_quality=jpeg_quality)
                    jpg_out = encode_jpeg(raw_out, max_side=max_side_eff, jpeg_quality=jpeg_quality)
                    if not jpg_in or not jpg_out:
                        st["bad_image"] += 1
                        continue

                    key = make_sample_key(stem, pair_index_global, prompt)
                    txt = (prompt.strip() + "\n").encode("utf-8")
                    buffer.append((key, jpg_in, jpg_out, txt))
                    pair_index_global += 1
                    st["kept"] += 1
                    total_kept += 1

                    if len(buffer) >= samples_per_shard:
                        flush_shard()
                        if max_shards and shard_idx >= max_shards:
                            per_file_stats.append(st)
                            summary = {
                                "dataset_root": str(dataset_root),
                                "stopped_early_max_shards": max_shards,
                                "shards_written": shard_idx,
                                "total_kept": total_kept,
                                "bucket": bucket_eff or None,
                                "shard_prefix": shard_prefix or None,
                                "per_parquet": per_file_stats + [st],
                            }
                            if (image_subdir or "").strip():
                                summary["image_subdir"] = image_subdir.strip()
                            write_json(report_path, summary)
                            print(f"Stopped (--max-shards). Report: {report_path}")
                            return summary

        per_file_stats.append(st)
        print(f"  kept {st['kept']} / pairs_declared {st['pairs_emitted']} / rows {st['rows']}")

    flush_shard()

    summary = {
        "dataset_root": str(dataset_root),
        "parquet_files": len(parquets),
        "shards_written": shard_idx,
        "total_kept": total_kept,
        "bucket": bucket_eff or None,
        "shard_prefix": shard_prefix or None,
        "subst_rules": list(subst_rules),
        "per_parquet": per_file_stats,
    }
    if (image_subdir or "").strip():
        summary["image_subdir"] = image_subdir.strip()
    write_json(report_path, summary)
    print(f"\nDone. Shards: {shard_idx}, samples: {total_kept}. Report: {report_path}")
    return summary


def main() -> None:
    args = parse_args()
    warn_bucket_zone(args.bucket, args.expected_zone)
    subst_rules = parse_substs(args.subst)
    work = Path(args.work_dir)
    report_path = Path(args.report_path) if args.report_path else work / "build_report.json"
    parquets_filter = parse_parquets_only(args.parquets_only)

    if args.full_snapshot_download:
        root = Path(args.dataset_root or "/dev/shm/imgedit_full").resolve()
        root.mkdir(parents=True, exist_ok=True)
        print(f"Full snapshot → {root}")
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(root),
            local_dir_use_symlinks=False,
        )
        dataset_root = root
    else:
        if not args.dataset_root:
            raise SystemExit("Set --dataset-root or use --full-snapshot-download.")
        dataset_root = Path(args.dataset_root).resolve()

    run_build(
        dataset_root=dataset_root,
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
        shard_prefix=args.shard_prefix,
        parquets_filter=parquets_filter,
        image_subdir=(args.image_subdir or "").strip(),
    )


if __name__ == "__main__":
    main()
