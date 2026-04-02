#!/usr/bin/env bash
# Process every ImgEdit parquet slice with explicit Hub archive globs.
# Requires: gcloud auth, HF access, stage1 parquets available to snapshot_download (or HF cache).
#
# Does not use `set -e`: one failed chunk does not stop the remainder.
# SKIPPED (exit 2 from stage2_chunk): audit below --min-resolve-rate, zero samples, etc.
#
# Skipped by omission (no chunk run): reference_replace_part1, reference_replace_part7

set -uo pipefail

BUCKET="${BUCKET:-gs://kmh-gcp-us-central1/data/imgedit}"
ZONE="${ZONE:-us-central1}"
LOG="${LOG:-/dev/shm/imgedit_run.log}"
SUMMARY="${SUMMARY:-/dev/shm/imgedit_summary.log}"

CNT_SUCCESS=0
CNT_SKIPPED=0
CNT_FAILED=0

exec > >(tee -a "${LOG}") 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Usage: run_chunk STEM [--flags for stage2_chunk.py...] PATTERN [PATTERN...]
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

  local code=0
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
    --shard-prefix "" \
    --min-resolve-rate 1.00 \
    --skip-existing \
    --delete-chunk-after \
    --delete-chunk-on-failure \
    || code=$?

  # Safety net: always free /dev/shm slice dirs (Python also cleans chunk on success/skip when flags set).
  rm -rf "/dev/shm/imgedit_chunk_${stem}"
  rm -rf "/dev/shm/imgedit_wds_${stem}"

  local dest="${BUCKET}/${stem}"
  if [[ "${code}" -eq 0 ]]; then
    echo "SUCCESS: ${stem} → ${dest}" | tee -a "${SUMMARY}"
    CNT_SUCCESS=$((CNT_SUCCESS + 1))
  elif [[ "${code}" -eq 2 ]]; then
    echo "SKIPPED: ${stem} → audit_below_threshold_or_zero_samples_or_unknown_schema (exit ${code})" | tee -a "${SUMMARY}"
    CNT_SKIPPED=$((CNT_SKIPPED + 1))
  else
    echo "FAILED: ${stem} → exit_code=${code} (unexpected or build error)" | tee -a "${SUMMARY}"
    CNT_FAILED=$((CNT_FAILED + 1))
  fi
}

echo "run_all_chunks.sh start $(date -Is)  BUCKET=${BUCKET}  ZONE=${ZONE}  SUMMARY=${SUMMARY}"
echo "# run_all_chunks start $(date -Is) BUCKET=${BUCKET}" >> "${SUMMARY}"

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
# run_chunk content_memory_part2 Multiturn/results_content_memory_part2.tar.split.*
# run_chunk content_understanding_part2 Multiturn/results_content_understanding_part2.tar.split.*
# run_chunk version_backtracking_part0 Multiturn/results_version_backtracking_part0.tar.split.*

echo ""
echo "========== TOTALS $(date -Is) =========="
echo "SUCCESS (uploaded): ${CNT_SUCCESS}"
echo "SKIPPED (audit / zero samples / exit 2): ${CNT_SKIPPED}"
echo "FAILED (other / exit ≠ 0,2): ${CNT_FAILED}"
{
  echo ""
  echo "========== TOTALS $(date -Is) =========="
  echo "SUCCESS (uploaded): ${CNT_SUCCESS}"
  echo "SKIPPED (audit / zero samples / exit 2): ${CNT_SKIPPED}"
  echo "FAILED (other): ${CNT_FAILED}"
} >> "${SUMMARY}"

echo "run_all_chunks.sh done $(date -Is)"
