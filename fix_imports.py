#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

def fix_imports(file_path):
    """Replace 'from main.' with 'from main.' in Python files."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace imports
    modified_content = re.sub(r'from purpose\.', 'from main.', content)
    modified_content = re.sub(r'import purpose\.', 'import main.', modified_content)
    
    # Only write to file if changes were made
    if content != modified_content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        return True
    return False

def process_directory(directory):
    """Process all Python files in the directory and its subdirectories."""
    modified_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                if fix_imports(file_path):
                    modified_files.append(file_path)
    
    return modified_files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "."
    
    modified_files = process_directory(target_dir)
    
    print(f"Fixed imports in {len(modified_files)} files:")
    for file in modified_files:
        print(f"  - {file}") 