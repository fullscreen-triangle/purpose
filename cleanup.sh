#!/bin/bash
# Cleanup script to remove old files after migration to package structure

echo "Cleaning up old files after migration to package structure..."

# Check if each file exists before attempting to remove it
if [ -f train_sprint_llm.py ]; then
  echo "Removing train_sprint_llm.py (migrated to purpose/examples/sprint/)"
  rm train_sprint_llm.py
fi

if [ -f process_content_data.py ]; then
  echo "Removing process_content_data.py (migrated to purpose/examples/sprint/)"
  rm process_content_data.py
fi

if [ -f run_pipeline.py ]; then
  echo "Removing run_pipeline.py (migrated to purpose/examples/sprint/)"
  rm run_pipeline.py
fi

if [ -f use_sprint_llm.py ]; then
  echo "Removing use_sprint_llm.py (migrated to purpose/inference/)"
  rm use_sprint_llm.py
fi

# Check for src directory (old structure)
if [ -d src ]; then
  echo "Warning: 'src' directory still exists. You may want to review and migrate any remaining files."
fi

# Check for domain_llm directory (old name)
if [ -d domain_llm ]; then
  echo "Warning: 'domain_llm' directory still exists. Run './migrate_to_purpose.sh' to migrate to the new package name."
fi

echo "Cleanup complete!"
echo "You can now use the purpose package with the following commands:"
echo "  - purpose process --data-dir content --output-dir data/processed"
echo "  - purpose train --data-dir data/processed --model-name gpt2 --output-name sprint_model"
echo "  - purpose generate --model-dir models/sprint_model --interactive --qa-mode"
echo ""
echo "Or you can run the full pipeline example:"
echo "  python -m purpose.examples.sprint.run_pipeline" 