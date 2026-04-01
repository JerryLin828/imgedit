#!/usr/bin/env python3
"""
Stage 2 — After a full local snapshot: strict WebDataset shards + upload to GCS.

Keeps a sample only when exactly one input path and one output path are declared,
both files exist, read successfully, and JPEG re-encode succeeds.
Optional --subst OLD:NEW only after you verified renames on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from imgedit_lib import (
    add_tar_bytes,
    detect_field_mapping,
    encode_jpeg,
    iter_imgedit_pairs_from_row,
    parquet_schema_columns,
    path_string_to_bytes,
    resolve_dataset_image_path,
    sanity_check_tar_triplet,
    upload_shard_noclobber,
    gcs_object_exists,
    write_json,
)

DEFAULT_REPO = "sysuyy/ImgEdit"


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
        help="gs://bucket/prefix — empty means keep .tar files local only.",
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


def main() -> None:
    args = parse_args()
    subst_rules = parse_substs(args.subst)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else work / "build_report.json"

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

    parquets = sorted(dataset_root.glob("Parquet/*.parquet"))
    if not parquets:
        raise SystemExit(f"No Parquet/*.parquet under {dataset_root}")

    max_side = None if args.max_side <= 0 else args.max_side
    bucket = (args.bucket or "").strip().rstrip("/")

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
        tar_name = f"shard-{shard_idx:05d}.tar"
        tar_path = work / tar_name
        gcs_uri = f"{bucket}/{tar_name}" if bucket else ""

        if args.skip_existing and gcs_uri and gcs_object_exists(gcs_uri):
            print(f"  skip (exists) {gcs_uri}")
            buffer.clear()
            shard_idx += 1
            return

        with tarfile.open(tar_path, "w") as tf:
            for key, j1, j2, txt in buffer:
                add_tar_bytes(tf, f"{key}.jpg", j1)
                add_tar_bytes(tf, f"{key}.jpg2", j2)
                add_tar_bytes(tf, f"{key}.txt", txt)

        sanity_check_tar_triplet(str(tar_path), n, decode_samples=args.sanity_decode)
        if bucket:
            upload_shard_noclobber(str(tar_path), bucket + "/")
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
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            st["rows"] += batch.num_rows
            for row in _row_dicts_from_batch(batch):
                for inp_rel, out_rel, prompt in iter_imgedit_pairs_from_row(
                    row, mode=mode, orig_col=orig_c or "", edit_col=edit_c or "", text_col=text_c
                ):
                    st["pairs_emitted"] += 1
                    if not (
                        resolve_dataset_image_path(
                            inp_rel,
                            dataset_root=dataset_root,
                            parquet_path=pq_path,
                            subst_rules=subst_rules,
                        )
                        and resolve_dataset_image_path(
                            out_rel,
                            dataset_root=dataset_root,
                            parquet_path=pq_path,
                            subst_rules=subst_rules,
                        )
                    ):
                        st["missing_file"] += 1
                        continue

                    raw_in = path_string_to_bytes(
                        inp_rel,
                        dataset_root=dataset_root,
                        parquet_path=pq_path,
                        subst_rules=subst_rules,
                    )
                    raw_out = path_string_to_bytes(
                        out_rel,
                        dataset_root=dataset_root,
                        parquet_path=pq_path,
                        subst_rules=subst_rules,
                    )
                    if not raw_in or not raw_out:
                        st["missing_file"] += 1
                        continue

                    jpg_in = encode_jpeg(raw_in, max_side=max_side, jpeg_quality=args.jpeg_quality)
                    jpg_out = encode_jpeg(raw_out, max_side=max_side, jpeg_quality=args.jpeg_quality)
                    if not jpg_in or not jpg_out:
                        st["bad_image"] += 1
                        continue

                    key = make_sample_key(stem, pair_index_global, prompt)
                    txt = (prompt.strip() + "\n").encode("utf-8")
                    buffer.append((key, jpg_in, jpg_out, txt))
                    pair_index_global += 1
                    st["kept"] += 1
                    total_kept += 1

                    if len(buffer) >= args.samples_per_shard:
                        flush_shard()
                        if args.max_shards and shard_idx >= args.max_shards:
                            per_file_stats.append(st)
                            write_json(
                                report_path,
                                {
                                    "dataset_root": str(dataset_root),
                                    "stopped_early_max_shards": args.max_shards,
                                    "shards_written": shard_idx,
                                    "total_kept": total_kept,
                                    "bucket": bucket or None,
                                    "per_parquet": per_file_stats + [st],
                                },
                            )
                            print(f"Stopped (--max-shards). Report: {report_path}")
                            return

        per_file_stats.append(st)
        print(f"  kept {st['kept']} / pairs_declared {st['pairs_emitted']} / rows {st['rows']}")

    flush_shard()

    write_json(
        report_path,
        {
            "dataset_root": str(dataset_root),
            "parquet_files": len(parquets),
            "shards_written": shard_idx,
            "total_kept": total_kept,
            "bucket": bucket or None,
            "subst_rules": list(subst_rules),
            "per_parquet": per_file_stats,
        },
    )
    print(f"\nDone. Shards: {shard_idx}, samples: {total_kept}. Report: {report_path}")


if __name__ == "__main__":
    main()
