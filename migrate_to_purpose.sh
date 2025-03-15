#!/bin/bash
# Migration script to help users transition from domain_llm to purpose

echo "Migrating from domain_llm to purpose..."

# Create purpose directory structure
mkdir -p purpose/{examples,inference,processing,trainer}

# Copy files from domain_llm to purpose
if [ -d "domain_llm" ]; then
  echo "Copying files from domain_llm to purpose..."
  cp -r domain_llm/* purpose/
  
  # Rename references in Python files
  echo "Updating imports in Python files..."
  find purpose -name "*.py" -type f -exec sed -i '' 's/from domain_llm/from purpose/g' {} \;
  find purpose -name "*.py" -type f -exec sed -i '' 's/import domain_llm/import purpose/g' {} \;
  
  # Rename references in example files
  echo "Updating example files..."
  find purpose/examples -name "*.py" -type f -exec sed -i '' 's/domain_llm/purpose/g' {} \;
  find purpose/examples -name "*.md" -type f -exec sed -i '' 's/domain_llm/purpose/g' {} \;
  find purpose/examples -name "README.md" -type f -exec sed -i '' 's/domain-llm/purpose/g' {} \;
  
  # Update references in markdown files
  echo "Updating documentation..."
  find purpose -name "*.md" -type f -exec sed -i '' 's/domain_llm/purpose/g' {} \;
  find purpose -name "*.md" -type f -exec sed -i '' 's/domain-llm/purpose/g' {} \;
  
  # Update file contents that might have domain_model references
  echo "Updating model references..."
  find purpose -name "*.py" -type f -exec sed -i '' 's/domain_model/purpose_model/g' {} \;
  find purpose -name "*.py" -type f -exec sed -i '' 's/domain_corpus/purpose_corpus/g' {} \;
  
  echo "Migration completed successfully!"
  echo ""
  echo "Next steps:"
  echo "1. Review the changes in the 'purpose' directory"
  echo "2. Update your import statements in any custom scripts"
  echo "3. Use the 'purpose' command instead of 'domain-llm' in your workflows"
  echo "4. Install the package with: pip install -e ."
else
  echo "Error: domain_llm directory not found. Make sure you're running this script from the root directory of the project."
  exit 1
fi 