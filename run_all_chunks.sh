#!/usr/bin/env bash
# Process every ImgEdit parquet slice with explicit Hub archive globs.
# Requires: gcloud auth, HF access, stage1 parquets available to snapshot_download (or HF cache).
#
# Skipped (no archives on Hub for strict WDS): reference_replace_part1, reference_replace_part7
#
# Optional: if remove_part0 fails audit (partial Hub data), re-run that stem with e.g.
#   --min-resolve-rate 0.753
# appended manually or extend run_chunk to accept extra python args.

set -euo pipefail

BUCKET="${BUCKET:-gs://kmh-gcp-us-central1/data/imgedit}"
ZONE="${ZONE:-us-central1}"
LOG="${LOG:-/dev/shm/imgedit_run.log}"

exec > >(tee -a "${LOG}") 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Usage: run_chunk STEM [--flags for stage2_chunk.py...] PATTERN [PATTERN...]
# Patterns are snapshot_download allow_patterns (e.g. Singleturn/results_*.tar.split.*).
# Flags must start with -- and are passed through (e.g. --fix-hybrid-compose-paths).
run_chunk() {
  local stem="$1"
  shift
  local allow=()
  local extras=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --* ]]; then
      extras+=("$1")
      shift
    else
      allow+=(--allow-pattern "$1")
      shift
    fi
  done
  echo ""
  echo "========== $(date -Is) ========== ${stem}"
  python stage2_chunk.py \
    --chunk-dir "/dev/shm/imgedit_chunk_${stem}" \
    --parquet "Parquet/${stem}.parquet" \
    "${allow[@]}" \
    "${extras[@]}" \
    --work-dir "/dev/shm/imgedit_wds_${stem}" \
    --bucket "${BUCKET}/${stem}" \
    --expected-zone "${ZONE}" \
    --skip-existing \
    --delete-chunk-after
}

echo "run_all_chunks.sh start $(date -Is)  BUCKET=${BUCKET}  ZONE=${ZONE}"

# --- Singleturn ---
run_chunk action_part1 Singleturn/action_part1.tar.split.*
run_chunk action_part2 Singleturn/action_part2.tar.split.*
run_chunk action_part3 Singleturn/action_part3.tar.split.*
run_chunk action_part4 Singleturn/action_part4.tar.split.*

run_chunk add_part0 Singleturn/results_add_laion_part0.tar.split.*
run_chunk add_part1 Singleturn/results_add_laion_part1.tar.split.*
run_chunk add_part4 Singleturn/results_add_laion_part4.tar.split.*
run_chunk add_part5 Singleturn/results_add_laion_part5.tar.split.*

run_chunk adjust_canny_part0 Singleturn/results_adjust_canny_laion_part0.tar.split.*
run_chunk adjust_canny_part2 Singleturn/results_adjust_canny_laion_part2.tar.split.*
run_chunk adjust_canny_part3 Singleturn/results_adjust_canny_laion_part3.tar.split.*
run_chunk adjust_canny_part4 Singleturn/results_adjust_canny_laion_part4.tar.split.*

run_chunk background_part0 Singleturn/results_background_laion_part0.tar.split.*
run_chunk background_part2 Singleturn/results_background_laion_part2.tar.split.*
run_chunk background_part3 Singleturn/results_background_laion_part3.tar.split.*
run_chunk background_part5 Singleturn/results_background_laion_part5.tar.split.*
run_chunk background_part7 Singleturn/results_background_laion_part7.tar.split.*

# Explicit compose→hybrid path subst (batch script does not rely on stem auto-detection)
run_chunk hybrid_part0 Singleturn/results_hybrid_part0.tar.split.* --fix-hybrid-compose-paths
run_chunk hybrid_part2 Singleturn/results_hybrid_part2.tar.split.* --fix-hybrid-compose-paths
run_chunk hybrid_part6 Singleturn/results_hybrid_part6.tar.split.* --fix-hybrid-compose-paths

run_chunk reference_extract_part1 Singleturn/results_extract_and_visualedit_part1.tar.split.*
run_chunk reference_extract_part7 Singleturn/results_extract_and_visualedit_part7.tar.split.*

run_chunk remove_part0 Singleturn/results_remove_part0.tar.split.*
run_chunk remove_part1 Singleturn/results_remove_laion_part1.tar.split.*
run_chunk remove_part4 Singleturn/results_remove_laion_part4.tar.split.*
run_chunk remove_part5 Singleturn/results_remove_laion_part5.tar.split.*

run_chunk replace_part0 Singleturn/results_replace_part0.tar.split.*
run_chunk replace_part1 Singleturn/results_replace_laion_part1.tar.split.*
run_chunk replace_part4 Singleturn/results_replace_laion_part4.tar.split.*
run_chunk replace_part5 Singleturn/results_replace_laion_part5.tar.split.*

run_chunk style_transfer Singleturn/results_style_transfer.tar.split.*
run_chunk style_transfer_part0 Singleturn/results_style_transfer_part0.tar.split.*

# --- Multiturn ---
run_chunk content_memory_part2 Multiturn/results_content_memory_part2.tar.split.*
run_chunk content_understanding_part2 Multiturn/results_content_understanding_part2.tar.split.*
run_chunk version_backtracking_part0 Multiturn/results_version_backtracking_part0.tar.split.*

# reference_replace_part1 / reference_replace_part7 — skipped (no matching strict-pair archives per project notes)

echo "run_all_chunks.sh done $(date -Is)"
