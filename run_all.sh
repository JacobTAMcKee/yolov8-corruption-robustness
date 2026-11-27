#!/usr/bin/env bash
# Runs the whole project in order with the default settings.
#
# Usage:
#   bash run_all.sh                    # everything
#   bash run_all.sh --skip-failures    # skip the slowest step (failure-case figures)
set -eo pipefail

SKIP_FAILURES=0
for arg in "$@"; do
    case "$arg" in
        --skip-failures) SKIP_FAILURES=1 ;;
        *) echo "Unknown option: $arg"; echo "Usage: bash run_all.sh [--skip-failures]"; exit 1 ;;
    esac
done

# Run one step: print its name, run it, report how long it took.
# Because of "set -e", the whole script stops if any step fails.
step() {
    local label="$1"; shift
    echo; echo "==> $label"
    local start=$SECONDS
    "$@"
    echo "    finished in $((SECONDS - start))s"
}

# 1. Setup, only if it has not been done yet
if [ ! -d venv ] || [ ! -f subset.yaml ]; then
    step "Setup: environment, COCO val2017, 500-image subset" bash setup.sh
fi
source venv/bin/activate

# 2. Fail early with a clear message if there is no CUDA GPU (the scripts use device=0)
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'No CUDA GPU found. See the README.')"

mkdir -p results

# 3. The pipeline, in order
step "Build the 12 corrupted image sets"            python make_corrupted.py
step "Corruption example figure"                    python check_corruptions.py
step "Evaluate yolov8n and yolov8s (13 sets each)"  python run_experiments.py
step "Plots and summary table"                      python make_plots.py
step "mAP50, precision, recall table"               python metrics_pct_of_clean.py
if [ "$SKIP_FAILURES" -eq 0 ]; then
    step "Failure-case figures"                     python find_failures.py
fi

echo; echo "All steps finished in $((SECONDS / 60)) min. Outputs are in results/."