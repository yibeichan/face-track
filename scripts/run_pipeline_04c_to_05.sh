#!/bin/bash

#SBATCH --job-name=refine_timestamps
#SBATCH --output=./logs/%x_%j_%a.out
#SBATCH --error=./logs/%x_%j_%a.err
#SBATCH --partition=mit_preemptable
#SBATCH --time=12:00:00
#SBATCH --array=1-7
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=4G
#SBATCH --mail-type=FAIL,END

# Batch processing script for refinement (04c) and timestamp generation (05).
# Runs on reviewed annotation output to produce per-frame character presence
# with face bounding box coordinates.
#
# Parallelism model: ONE JOB PER SEASON (7 array tasks for s1-s7).
# Each task grabs every episode listed for its season in data/episode_id.txt
# and processes them sequentially on a single GPU allocation. This trades
# fine-grained parallelism for far less scheduler churn and one warm-up per
# season rather than per episode.
#
# Annotation lookup:
#     output/reviewed_output/s{N}/<episode_id>_annotations.json
#
# Prerequisites:
#   - Stages 01-04a already completed for all episodes in the target season
#   - Reviewed annotations present under output/reviewed_output/s{N}/
#
# Season -> array task mapping:
#   SLURM_ARRAY_TASK_ID = season number (1-7)
#   e.g. --array=3 runs all s3 episodes in a single job
#
# Episode counts per season (for time-budget sanity):
#   s1: 48 episodes    s4: 48 episodes
#   s2: 48 episodes    s5: 48 episodes
#   s3: 50 episodes    s6: 50 episodes
#                       s7: 49 episodes
#
# Usage:
#   # Process all reviewed seasons s1-s7 (7 parallel jobs, default):
#   sbatch scripts/run_pipeline_04c_to_05.sh
#
#   # Process a single season:
#   sbatch --array=1 scripts/run_pipeline_04c_to_05.sh   # s1 only
#   sbatch --array=6 scripts/run_pipeline_04c_to_05.sh   # s6 only
#
#   # Process a subset of seasons:
#   sbatch --array=1,3,5 scripts/run_pipeline_04c_to_05.sh
#
#   # Run locally for testing a single episode (bypasses the season loop):
#   ./scripts/run_pipeline_04c_to_05.sh friends_s01e01a
#
# Failure behaviour:
#   Episode-level failures are logged but do NOT abort the season loop - the
#   job continues and reports a summary (pass/fail counts + failed episode
#   list) at the end. The job exits non-zero if any episode failed, so SLURM
#   still marks the task as failed and the FAIL mail goes out.
#
# IMPORTANT: Submit from the project root directory:
#   cd /home/yibei/char-tracker
#   sbatch scripts/run_pipeline_04c_to_05.sh

# Load cuDNN module for onnxruntime-gpu (required for libcudnn.so.9)
module load cudnn/9.8.0.87-cuda12

# Activate uv virtual environment
source "${SLURM_SUBMIT_DIR:-.}/.venv/bin/activate"

# Set up paths
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPTS_DIR="$REPO_ROOT/scripts"
TASK_FILE="$REPO_ROOT/data/episode_id.txt"
# Annotation root - per-episode subdir (s1/, s2/, ...) is appended below,
# once the season is parsed from the episode ID.
ANNOTATION_ROOT="$REPO_ROOT/output/reviewed_output"
LOG_DIR="$REPO_ROOT/logs"

mkdir -p "$LOG_DIR"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: Task file not found: $TASK_FILE"
    exit 1
fi

# Determine the list of episodes to process.
# SLURM mode: task ID is the season number; loop all episodes for that season.
# Local mode:  single episode ID passed as $1 (testing path).
EPISODES=()
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEASON="$SLURM_ARRAY_TASK_ID"
    # Zero-pad for grep against "friends_sNN..." format
    SEASON_PADDED=$(printf "%02d" "$SEASON")
    while IFS= read -r ep; do
        EPISODES+=("$ep")
    done < <(grep "^friends_s${SEASON_PADDED}e" "$TASK_FILE")

    if [ ${#EPISODES[@]} -eq 0 ]; then
        echo "ERROR: No episodes found for season $SEASON (pattern: friends_s${SEASON_PADDED}e)"
        exit 1
    fi
else
    # Running locally - accept a single episode ID
    if [ -z "${1:-}" ]; then
        echo "=========================================="
        echo "ERROR: Not running via SLURM and no episode specified"
        echo ""
        echo "Usage (SLURM - recommended):"
        echo "  sbatch scripts/run_pipeline_04c_to_05.sh              # all s1-s7"
        echo "  sbatch --array=1 scripts/run_pipeline_04c_to_05.sh    # s1 only"
        echo "  sbatch --array=1,3,5 scripts/run_pipeline_04c_to_05.sh"
        echo ""
        echo "Usage (Local - for testing single episode):"
        echo "  ./scripts/run_pipeline_04c_to_05.sh friends_s01e01a"
        echo "=========================================="
        exit 1
    fi
    if ! grep -q "^${1}$" "$TASK_FILE"; then
        echo "ERROR: Episode '$1' not found in $TASK_FILE"
        exit 1
    fi
    EPISODES=("$1")
    # Parse season from the episode ID so log output is consistent
    if [[ "${EPISODES[0]}" =~ ^friends_s([0-9]+)e ]]; then
        SEASON=$((10#${BASH_REMATCH[1]}))
    else
        echo "ERROR: Could not parse season from episode ID: ${EPISODES[0]}"
        exit 1
    fi
fi

ANNOTATION_DIR="$ANNOTATION_ROOT/s${SEASON}"

echo "=========================================="
echo "REFINEMENT + TIMESTAMP GENERATION"
echo "=========================================="
echo "SLURM Array Job ID:  ${SLURM_ARRAY_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-local}"
echo "Season:              s${SEASON}"
echo "Episodes:            ${#EPISODES[@]}"
echo "Annotation dir:      $ANNOTATION_DIR"
echo "Node:                $(hostname)"
echo "=========================================="
echo ""

cd "$SCRIPTS_DIR" || exit 1

# Per-episode loop. Failures are tallied but do not abort the job so that
# transient issues on one episode don't block the remaining 40+.
SUCCEEDED=()
FAILED=()
START_TIME=$(date +%s)

for EPISODE_ID in "${EPISODES[@]}"; do
    EP_START=$(date +%s)
    ANNOTATION_FILE="$ANNOTATION_DIR/${EPISODE_ID}_annotations.json"

    echo "=========================================="
    echo "[$((${#SUCCEEDED[@]} + ${#FAILED[@]} + 1))/${#EPISODES[@]}] $EPISODE_ID"
    echo "=========================================="

    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "SKIP: annotation file not found: $ANNOTATION_FILE"
        FAILED+=("$EPISODE_ID (missing annotation)")
        echo ""
        continue
    fi

    # Stage 04c: Refine clustering with annotations
    echo "--- Stage 04c: Refine with annotations ---"
    python 04c_refine_with_annotations.py "$EPISODE_ID" "$ANNOTATION_FILE"
    EXIT_04C=$?

    if [ $EXIT_04C -ne 0 ]; then
        echo "FAILED: Stage 04c for $EPISODE_ID (exit $EXIT_04C)"
        FAILED+=("$EPISODE_ID (04c exit $EXIT_04C)")
        echo ""
        continue
    fi

    # Stage 05: Generate character timestamps
    echo ""
    echo "--- Stage 05: Generate character timestamps ---"
    python 05_generate_character_timestamps.py "$EPISODE_ID"
    EXIT_05=$?

    if [ $EXIT_05 -ne 0 ]; then
        echo "FAILED: Stage 05 for $EPISODE_ID (exit $EXIT_05)"
        FAILED+=("$EPISODE_ID (05 exit $EXIT_05)")
        echo ""
        continue
    fi

    EP_ELAPSED=$(( $(date +%s) - EP_START ))
    echo ""
    echo "OK: $EPISODE_ID (${EP_ELAPSED}s)"
    SUCCEEDED+=("$EPISODE_ID")
    echo ""
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))

echo "=========================================="
echo "SEASON s${SEASON} SUMMARY"
echo "=========================================="
echo "Total elapsed:  ${TOTAL_ELAPSED}s"
echo "Succeeded:      ${#SUCCEEDED[@]} / ${#EPISODES[@]}"
echo "Failed:         ${#FAILED[@]} / ${#EPISODES[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed episodes:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "=========================================="

# Non-zero exit if anything failed, so SLURM reports the task as failed and
# the FAIL mail fires.
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
