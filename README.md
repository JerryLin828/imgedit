# Preparing ImgEdit on GCS Bucket

Two-stage conversion from Hugging Face dataset `sysuyy/ImgEdit` into WebDataset shards. **Only rows where both images exist on disk and decode cleanly are kept** (correctness over raw row count). Design mirrors `deepfusion/dataset/docs/magicbrush.md` and `upload_magicbrush_webdataset.py`: **`gcloud storage cp -n`**, `/dev/shm` scratch, optional **zone check** on the bucket path, and **disk usage logging** during chunk processing.

- **Source format:** `Parquet/*.parquet` (lists of paths + `prompt`; multiturn uses a `data` column). Binary blobs live under `Singleturn/` and `Multiturn/` as `results_*.tar.split.*` and sometimes whole `.tar` archives; **`stage2_chunk.py`** merges splits, extracts standalone tars, then audits path resolution.
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
- `imgedit/stage2_chunk.py` — HF slice download + build + optional chunk dir delete
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
3. Packs samples into `shard-XXXXX.tar` (width 5 by default), or `{shard-prefix}-shard-XXXXX.tar` when `--shard-prefix` is set (for chunked runs).
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

**Stage 2** after full download + extract (match `<zone>` to your VM; canonical prefix in code is `DEFAULT_BUCKET` in `stage2_build_webdataset.py`, same `gs://kmh-gcp-…/data/<name>` layout as MagicBrush):

```bash
python stage2_build_webdataset.py \
  --dataset-root /path/to/imgedit_snapshot \
  --work-dir /dev/shm/imgedit_wds \
  --bucket gs://kmh-gcp-<zone>/data/imgedit \
  --expected-zone <zone> \
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

## Chunk workflow (one parquet + tars → validate → upload → delete)

Full ImgEdit is multi‑terabyte on Hugging Face. With **limited local disk**, run **one chunk at a time**: download one parquet (via `--parquet`) plus the `Singleturn` / `Multiturn` globs for that slice, **merge `*.tar.split.*` in order and extract** inside `--chunk-dir`, **audit** that strict parquet paths resolve on disk, build shards (default **`--shard-prefix` empty** = `shard-00000.tar` like MagicBrush; use a **per-chunk GCS subdir** or set `--shard-prefix` for a flat bucket), upload, then **`--delete-chunk-after`** to remove the chunk.

**Single command** (example — adjust `--allow-pattern` to match [the repo](https://huggingface.co/datasets/sysuyy/ImgEdit/tree/main)):

```bash
cd imgedit
python stage2_chunk.py \
  --chunk-dir /dev/shm/imgedit_chunk_remove_p0 \
  --parquet Parquet/remove_part0.parquet \
  --allow-pattern "Singleturn/results_remove*.tar.split.*" \
  --work-dir /dev/shm/imgedit_wds \
  --bucket gs://kmh-gcp-<zone>/data/imgedit \
  --expected-zone <zone> \
  --skip-existing \
  --delete-chunk-after
```

This writes `audit_<parquet-stem>.json` (includes `subst_rules_applied`) before build; the run **fails** if `resolve_rate` on strict pairs is below `--min-resolve-rate` (default `1.0`), or if **no samples** pass path + decode + JPEG checks (`total_kept == 0`), so bad slices are not treated as success. Lower the audit bar only with `--min-resolve-rate`; override the sample check only with `--allow-empty-output`.

**Hybrid / compose naming:** `hybrid_part*.parquet` rows use path roots like `results_compose_part0` / `results_compose_part6_fix` while Hub file names use **`results_hybrid`**. `stage2_chunk.py` **turns on** `results_compose→results_hybrid` automatically for stems `hybrid_part*` (override with **`--no-auto-hybrid-compose`**). You still must pass **`--allow-pattern`** globs that match the **`results_hybrid*`** tar/split names on Hugging Face.

With `--delete-chunk-after`, the audit file is **copied** next to the build report before the chunk directory is removed.

### Reading `stage1_explore.log` for chunk runs

Typical lines from a full Stage 1 pass and how Stage 2 treats them:

| Log pattern | Meaning | `stage2_chunk.py` note |
|-------------|---------|-------------------------|
| `action_*.parquet` + *basename-only paths* | Paths are filenames only; Hub file list check is N/A | Resolver looks under `Singleturn/` / `Multiturn/` / dataset root; ensure your tars extract so those basenames exist there. |
| `hybrid_part*.parquet` + `roots not in hub names: ['results_compose_…']` | Parquet uses **compose** path prefix; Hub ships **hybrid** blobs | Compose→hybrid subst **auto** for `hybrid_part*` stems; download `results_hybrid*` patterns. |
| `reference_replace_*.parquet` + `strict_pairs=0` | Rows have multiple reference inputs (not one-in/one-out) | Current pipeline **skips** all rows; expect audit failure unless `--allow-zero-strict-pairs`, then **zero** samples unless you change pairing rules. |
| `reference_extract_*.parquet` + missing `results_extract_ref_*` in hub names | String mismatch vs published tar/tree names | Inspect Hub tree; may need **`--subst OLD:NEW`** after you confirm the real prefix. |
| `style_transfer_part0` + long root like `results_style_transfer_part0_cap36472` | Coarse stage1 substring check | Compare to actual extracted paths; add **`--subst`** if parquet strings don’t match on-disk dirs. |
| `strict_pairs` **>** `rows` (e.g. content / multiturn-style) | Several strict pairs emitted per row | Normal; Stage 2 iterates every pair. |

**Optional:** `--download-only` then re-run with `--skip-download` if you want a two-step handoff. **`--no-extract`** if you already merged/extracted manually. **`--skip-audit`** only for emergencies.

With an empty `--shard-prefix`, shards look like `shard-00000.tar`; **`run_all_chunks.sh`** uses **`--bucket $BUCKET/$STEM`** so each parquet lands under its own prefix. For a **flat** bucket, set **`--shard-prefix`** (e.g. parquet stem) so names do not collide.

You can also drive **`stage2_build_webdataset.py`** alone with `--dataset-root` pointing at a partial tree and **`--shard-prefix`** + **`--parquets-only`** if you manage downloads yourself.

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
- `--shard-prefix` — prefix for shard filenames (avoids GCS collisions between chunk runs).
- `--parquets-only` — comma-separated parquet basenames to process.
- `--expected-zone` — if set, log a warning when this substring is missing from `--bucket` (catches wrong-region buckets).

**Stage 2 chunk helper (`stage2_chunk.py`)**

- `--parquet` — repo-relative path, e.g. `Parquet/remove_part0.parquet` (always included in the download set).
- `--chunk-dir` — `local_dir` for HF download and `--dataset-root` for the build.
- `--allow-pattern` — repeat; extra `snapshot_download` patterns (image `*.tar` / `*.tar.split.*` globs).
- `--download-only` — fetch slice only; exit before extract / audit / build.
- `--skip-download` — chunk dir already populated.
- `--no-extract` — skip automatic extract (no `*.tar.split.*` merge, no standalone `.tar` unpack).
- `--delete-chunk-after` — after a successful build, remove `--chunk-dir`.
- `--shard-prefix` — defaults to empty (`shard-NNNNN.tar`); set explicitly when using one shared GCS prefix without subdirs.
- `--min-resolve-rate` — audit threshold on strict pairs (default `1.0`).
- `--allow-zero-strict-pairs` — allow parquets with no strict pairs (rare).
- `--skip-audit` — disable parquet↔disk check (not recommended).
- `--expected-zone` — same bucket/region sanity check as MagicBrush (`warn_bucket_zone`).
- `--fix-hybrid-compose-paths` — force compose→hybrid path subst (also **auto** for parquet stem `hybrid_part*` unless `--no-auto-hybrid-compose`).
- `--no-auto-hybrid-compose` — disable that auto behavior for `hybrid_part*.parquet`.
- `--allow-empty-output` — allow success when zero samples pass validation (debug only).

`stage2_chunk.py` logs **pre-download / post-extract / pre-build** disk usage like the MagicBrush uploader. There is **no** MagicBrush-only image cleanup (`--cleanup-corner`). Optional path fixes are **opt-in** via `--subst`.

---

## Differences vs MagicBrush (dataset shape)

| | MagicBrush (`deepfusion`) | ImgEdit (`imgedit/`) |
|---|---------------------------|------------------------|
| Parquet rows | Embedded image structs / bytes | Nested **path lists** → files under `Singleturn/` / `Multiturn/` |
| HF download | `hf_hub_download` per parquet | `snapshot_download` **allow_patterns** + tar **split merge** + optional whole **`.tar`** extract |
| Shards | Often one parquet → one `shard-XXXX.tar` | Many samples per shard (`--samples-per-shard`); optional **`--shard-prefix`** per chunk |
| Triplet layout | `{key}.jpg` / `.jpg2` / `.txt` | Same (shared `sanity_check_tar_triplet` / `gcloud storage cp -n` pattern) |

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
