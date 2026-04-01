# Preparing ImgEdit on GCS Bucket

Two-stage conversion from Hugging Face dataset `sysuyy/ImgEdit` into WebDataset shards. **Only rows where both images exist on disk and decode cleanly are kept** (correctness over raw row count).

- **Source format:** `Parquet/*.parquet` (lists of paths + `prompt`; multiturn uses a `data` column). Binary blobs live under `Singleturn/` and `Multiturn/` as `results_*.tar.split.*` chunks; you must **concatenate and extract** them so parquet paths resolve.
- **Processing strategy:**
  - **Stage 1:** small download (parquets + README + score JSON) and a JSON report (strict pair counts, coarse Hub prefix checks).
  - **Stage 2:** walk the full local tree, stream parquets, pack **validated** samples into `.tar` shards, upload with **`gcloud storage cp -n`**.
- **Target format:** `shard-00000.tar`, `shard-00001.tar`, … under GCS (several parquets may contribute to one shard; see `--samples-per-shard`).
- **Default local paths** use `/dev/shm`:
  - stage 1 snapshot: `/dev/shm/imgedit_stage1`
  - stage 2 work (shards before upload): `/dev/shm/imgedit_wds`

## Dependencies

```bash
pip install -r requirements.txt

# Authenticate with GCS
gcloud auth activate-service-account --key-file <path-to-key>.json
```

Optional: set `HF_TOKEN` if your environment needs authenticated Hugging Face access.

## Scripts

- `imgedit/stage1_explore.py` — light snapshot + `exploration_report.json`
- `imgedit/stage2_build_webdataset.py` — WebDataset build + optional GCS upload
- `imgedit/imgedit_lib.py` — path resolution, schema detection, tar/GCS helpers

### What it does

**Stage 1**

1. Optionally downloads only `Parquet/**`, `README.md`, `all_dataset_gpt_score.json`, `.gitattributes` (no huge image splits).
2. Scans every parquet in batch order and counts **strict** pairs: exactly one input path and one output path per edit (rows with multiple reference images are skipped).
3. Records each `results_*` path prefix and whether that substring appears **anywhere** in the Hub file list (rough signal for “is this tree published?”).
4. Writes `exploration_report.json` (default: under `--local-dir`).
5. Optional **`--log-file`**: append a text log of the run (stdout still shows the same lines).

**Stage 2**

1. Expects `--dataset-root` to contain `Parquet/` and **extracted** image files reachable from those paths (or pass `--full-snapshot-download` once; very large).
2. Reads rows in batches; for each strict pair, resolves both files under the dataset root (`Singleturn/`, `Multiturn/`, etc.), re-encodes as JPEG, drops anything missing or corrupt.
3. Packs samples into `shard-XXXXX.tar` (width 5 by default).
4. Runs sanity checks on each tar:
   - required suffixes per key (`jpg`, `jpg2`, `txt`)
   - counts match (`jpg == jpg2 == txt == samples_in_shard`)
   - decode a few samples (`--sanity-decode`) as images + UTF-8 text
5. Uploads each shard with `gcloud storage cp -n` when `--bucket` is set, then deletes the local tar after success.
6. Writes `build_report.json` (default: under `--work-dir`) with per-parquet kept / missing / bad counts.

## Quick Start

**Stage 1** (run first on the remote; low disk):

```bash
cd imgedit
python stage1_explore.py \
  --local-dir /dev/shm/imgedit_stage1 \
  --report-path /dev/shm/imgedit_stage1/exploration_report.json \
  --log-file /kmh-nfs-ssd-us-mount/code/<you>/imgedit/stage1_explore.log
```

Hub-only probe (no parquet download):

```bash
python stage1_explore.py --no-download --local-dir /dev/shm/imgedit_stage1 \
  --log-file /kmh-nfs-ssd-us-mount/code/<you>/imgedit/stage1_hub_only.log
```

**Stage 2** after full download + extract (match `<zone>` to your VM; same pattern as other `deepfusion/dataset` buckets):

```bash
python stage2_build_webdataset.py \
  --dataset-root /path/to/imgedit_snapshot \
  --work-dir /dev/shm/imgedit_wds \
  --bucket gs://kmh-gcp-<zone>/data/imgedit \
  --skip-existing
```

Smoke test (one shard, then stop):

```bash
python stage2_build_webdataset.py \
  --dataset-root /path/to/imgedit_snapshot \
  --work-dir /dev/shm/imgedit_wds \
  --bucket gs://kmh-gcp-<zone>/data/imgedit \
  --skip-existing \
  --max-shards 1
```

Run long jobs in `tmux`/`screen`.

## Key Args

**Stage 1 (`stage1_explore.py`)**

- `--repo` — HF dataset id (default `sysuyy/ImgEdit`).
- `--local-dir` — partial snapshot + default report location.
- `--no-download` — only call the Hub API; skip parquet analysis on disk.
- `--report-path` — override JSON report path.
- `--log-file` — append human-readable run log (UTF-8); parent dirs created; JSON report also records `log_file` when set.

**Stage 2 (`stage2_build_webdataset.py`)**

- `--dataset-root` — snapshot root with `Parquet/` and extracted images (required unless `--full-snapshot-download`).
- `--full-snapshot-download` — download the entire repo into `--dataset-root` (huge).
- `--work-dir` — where `shard-*.tar` are built (default `/dev/shm/imgedit_wds`).
- `--bucket` — `gs://.../prefix`; if omitted, shards stay local.
- `--skip-existing` — skip upload when `gs://…/<shard>.tar` already exists (resume-style).
- `--samples-per-shard` — shard size (default `1000`).
- `--batch-size` — PyArrow batch read size (default `256`).
- `--jpeg-quality` — JPEG quality (default `95`).
- `--max-side` — longest edge cap before JPEG (`0` = no resize).
- `--sanity-decode` — how many samples to decode per tar after write (default `3`).
- `--subst OLD:NEW` — optional verified path substring replace; repeatable; use only after checking real files on disk.
- `--max-shards` — stop after N shards (debug).
- `--report-path` — override `build_report.json` location.

There is **no** MagicBrush-only flags here (`--cleanup-corner`, `--dry-run`, per-parquet shard layout). Optional path fixes are **opt-in** via `--subst`.

## Monitoring & Resume

- List uploaded shards:

```bash
gcloud storage ls gs://kmh-gcp-<zone>/data/imgedit/shard-*.tar
```

- Size under the prefix:

```bash
gsutil du -s gs://kmh-gcp-<zone>/data/imgedit/
```

- **Resume:** use the same `--dataset-root` and parquet ordering; with `--skip-existing`, stage 2 skips shards that already exist on GCS. Shard indices advance in a single run; for complicated partial reruns, coordinate with your team or rerun from a clean manifest strategy.
- **Region:** set `<zone>` to the **same region as your VM** (see other `deepfusion/dataset/docs/*.md`) to avoid slow uploads and egress.
- **Disk:** if `/dev/shm` is too small, point `--local-dir` / `--work-dir` at an agreed SSD path.

## Output Layout

Each sample key contains 3 files:

- `{key}.jpg` (source / input image)
- `{key}.jpg2` (edited / target image)
- `{key}.txt` (instruction + trailing newline)

Note: with plain `webdataset.decode("pil")`, `jpg` is decoded automatically, while `jpg2` may remain bytes unless you add a custom decode step.

## License

ImgEdit (dataset) is Apache-2.0. This tooling is for your internal pipeline.
