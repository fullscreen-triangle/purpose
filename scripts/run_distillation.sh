#!/bin/bash
# run_distillation.sh
# 
# This script runs the distillation process after applying the necessary patches.
# Usage:
#   ./scripts/run_distillation.sh [arguments for run_distillation.py]
#
# Example:
#   ./scripts/run_distillation.sh --papers-dir content/papers --model-name distilgpt2

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Setup Python path
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Apply patches first
echo "Applying compatibility patches..."
python "$SCRIPT_DIR/patch_hub.py"

# Run the distillation script with all arguments passed to this script
echo "Running distillation process..."
python "$SCRIPT_DIR/run_distillation.py" "$@" 