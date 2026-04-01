# ImgEdit → WebDataset (two stages)

**Principle:** only keep samples where both images are really on disk and decode cleanly. Prefer correctness over count.

Run on a remote machine with Hugging Face access, a fast working directory (e.g. `/dev/shm`), and later `gcloud` for uploads.

---

## Setup

```bash
cd new
pip install -r requirements.txt
# Auth for stage 2 uploads:
gcloud auth activate-service-account --key-file <key>.json   # or `gcloud auth login`
```

---

## Stage 1 — Light snapshot & report (do this first)

Downloads **only** `Parquet/**`, `README.md`, and `all_dataset_gpt_score.json` (not the huge image splits).

- Reads every parquet row.
- Counts **strict** edit pairs: exactly one input path and one output path (multi-image rows are skipped).
- For each path prefix like `results_remove`, checks whether that string appears **anywhere** in the Hugging Face file list (coarse signal for “is this blob family published?”).

```bash
python stage1_explore.py --local-dir /dev/shm/imgedit_stage1
```

Outputs `exploration_report.json` next to the snapshot. Use it to see which parquet files will mostly fail before you move terabytes.

Options:

- `--no-download` — Hub API only (no parquet scan).
- `--report-path /path/to/report.json`

---

## Stage 2 — Full tree, WebDataset, and Google Cloud Storage (when you are ready)

You need the **full** dataset: run `huggingface-cli download` / `snapshot_download` without `allow_patterns`, **concatenate** each `results_*` split set, **extract** archives so paths in the parquets resolve under your root.

Then build shards (each sample: `{key}.jpg`, `{key}.jpg2`, `{key}.txt`) and upload:

```bash
python stage2_build_webdataset.py \
  --dataset-root /path/to/imgedit_snapshot \
  --work-dir /dev/shm/imgedit_wds \
  --bucket gs://YOUR_BUCKET/data/imgedit \
  --skip-existing
```

- **No `--subst`** by default. Add only after you prove a rename on disk, e.g.  
  `--subst results_compose_part0:results_hybrid_part0`
- `--full-snapshot-download` — optional; downloads **everything** into `--dataset-root` (very large). Usually you download once by other means and pass `--dataset-root` only.
- `--max-shards N` — smoke test.
- Report: `build_report.json` in `--work-dir` (override with `--report-path).

---

## Files

| File | Role |
|------|------|
| `imgedit_lib.py` | Paths, schema detection, strict pair iteration, tar/GCS helpers |
| `stage1_explore.py` | Small snapshot + JSON report |
| `stage2_build_webdataset.py` | Strict shards + optional GCS upload |

---

## License

The ImgEdit dataset is Apache-2.0. This tooling is for your internal pipeline.
