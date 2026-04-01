#!/usr/bin/env python3
"""
Stage 1 — Light snapshot + report (run on a remote machine with modest disk).

Downloads only metadata-rich pieces of sysuyy/ImgEdit (parquets + README + small JSON),
then summarizes shapes and checks whether path prefixes in the parquets appear anywhere
in the Hugging Face file list (a coarse "is this tree published?" signal).

Does NOT download Singleturn/Multiturn image splits. Use a full snapshot in stage 2 when ready.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from imgedit_lib import (
    detect_field_mapping,
    iter_imgedit_pairs_from_row,
    parquet_schema_columns,
    summarize_parquet,
    write_json,
)

DEFAULT_REPO = "sysuyy/ImgEdit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: light ImgEdit snapshot + exploration report.")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument(
        "--local-dir",
        default="/dev/shm/imgedit_stage1",
        help="Where to place the partial snapshot (use fast disk on remote).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Only query the Hub API and skip snapshot_download (no parquet scan on disk).",
    )
    p.add_argument(
        "--report-path",
        default="",
        help="Write JSON report here (default: <local-dir>/exploration_report.json).",
    )
    p.add_argument(
        "--log-file",
        default="",
        help="Append human-readable run log to this file (console still prints). Parent dirs are created.",
    )
    return p.parse_args()


def setup_logging(log_file: Path | None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _row_dicts_from_batch(batch) -> List[Dict[str, Any]]:
    d = batch.to_pydict()
    n = batch.num_rows
    keys = list(d.keys())
    return [{k: d[k][i] for k in keys} for i in range(n)]


def root_prefix(rel: str) -> str:
    parts = rel.strip("/").split("/")
    return parts[0] if parts else ""


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def is_basename_only_path(rel: str) -> bool:
    """Paths with no directory component (common in action_* parquets). Hub file list cannot validate these."""
    s = rel.strip("/")
    return "/" not in s and s.lower().endswith(_IMAGE_SUFFIXES)


def hub_check_meaningful_for_root(root: str) -> bool:
    if not root:
        return False
    if is_basename_only_path(root):
        return False
    return True


def analyze_parquet_file(path: Path, api_files: List[str]) -> Dict[str, Any]:
    """Scan one parquet: schema, strict pair counts, path roots, Hub name presence."""
    cols = parquet_schema_columns(path)
    mapping = detect_field_mapping(cols)
    mode = mapping[3]
    orig_c, edit_c, text_c = mapping[0], mapping[1], mapping[2]

    hub_joined = "\n".join(api_files)

    stats: Dict[str, Any] = {
        "file": str(path.name),
        "detected_mode": mode,
        "rows_total": 0,
        "strict_pairs_found": 0,
        "rows_with_zero_strict_pairs": 0,
        "rows_with_multiple_strict_pairs": 0,
        "path_roots": defaultdict(int),
        "roots_missing_in_hub_filenames": [],
        "note_basename_only_paths": False,
    }

    roots_counter: DefaultDict[str, int] = defaultdict(int)

    if mode == "unknown":
        stats["error"] = "unknown_schema"
        stats.update(summarize_parquet(path))
        return stats

    pf = pq.ParquetFile(path)
    read_cols = set(cols)
    target_cols = list(read_cols)

    for batch in pf.iter_batches(batch_size=2048, columns=target_cols):
        stats["rows_total"] += batch.num_rows
        for row in _row_dicts_from_batch(batch):
            n_pairs = 0
            for inp_rel, out_rel, _ in iter_imgedit_pairs_from_row(
                row, mode=mode, orig_col=orig_c or "", edit_col=edit_c or "", text_col=text_c
            ):
                n_pairs += 1
                for rel in (inp_rel, out_rel):
                    r = root_prefix(rel)
                    if r:
                        roots_counter[r] += 1
                        if is_basename_only_path(rel):
                            stats["note_basename_only_paths"] = True
            stats["strict_pairs_found"] += n_pairs
            if n_pairs == 0:
                stats["rows_with_zero_strict_pairs"] += 1
            elif n_pairs > 1:
                stats["rows_with_multiple_strict_pairs"] += 1

    stats["path_roots"] = dict(sorted(roots_counter.items(), key=lambda x: -x[1]))

    missing_roots: List[str] = []
    for root in stats["path_roots"].keys():
        if not hub_check_meaningful_for_root(root):
            continue
        if root and root not in hub_joined:
            missing_roots.append(root)
    stats["roots_missing_in_hub_filenames"] = sorted(missing_roots)

    return stats


def main() -> None:
    args = parse_args()
    local = Path(args.local_dir).resolve()
    report_path = Path(args.report_path) if args.report_path else local / "exploration_report.json"
    log_file_path: Path | None = Path(args.log_file).resolve() if args.log_file.strip() else None
    setup_logging(log_file_path)
    log = logging.getLogger(__name__)

    api = HfApi()
    api_files = api.list_repo_files(args.repo, repo_type="dataset")

    report: Dict[str, Any] = {
        "repo": args.repo,
        "stage": 1,
        "local_dir": str(local),
        "hub_file_count": len(api_files),
        "parquet_analysis": [],
    }
    if log_file_path is not None:
        report["log_file"] = str(log_file_path)

    if args.no_download:
        report["note"] = "Skipped download; parquet_analysis empty. Re-run without --no-download."
        write_json(report_path, report)
        log.info("Wrote %s (%s)", report_path, report["note"])
        return

    local.mkdir(parents=True, exist_ok=True)
    log.info("Partial snapshot → %s", local)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(local),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "Parquet/**",
            "README.md",
            "all_dataset_gpt_score.json",
            ".gitattributes",
        ],
    )

    parquets = sorted(local.glob("Parquet/*.parquet"))
    if not parquets:
        parquets = sorted(local.rglob("*.parquet"))

    total_rows = 0
    total_pairs = 0

    for pq_path in parquets:
        one = analyze_parquet_file(pq_path, api_files)
        report["parquet_analysis"].append(one)
        total_rows += one.get("rows_total", 0)
        total_pairs += one.get("strict_pairs_found", 0)
        bb = one.get("note_basename_only_paths")
        miss = one.get("roots_missing_in_hub_filenames") or []
        flag = f" ⚠ roots not in hub names: {miss[:3]}…" if miss else ""
        bnote = " (basename-only paths — hub list cannot validate; OK for action_* if tars extract to Singleturn/)" if bb else ""
        log.info(
            "  %s: rows=%s strict_pairs=%s%s%s",
            pq_path.name,
            one.get("rows_total"),
            one.get("strict_pairs_found"),
            bnote,
            flag,
        )

    report["totals"] = {
        "parquet_files": len(parquets),
        "rows_total": total_rows,
        "strict_pairs_total": total_pairs,
    }

    write_json(report_path, report)
    log.info(
        "Summary: %d parquet files, %d rows, %d strict single-pair extractions",
        len(parquets),
        total_rows,
        total_pairs,
    )
    log.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
