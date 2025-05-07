#!/usr/bin/env python3
"""
Directly patch the transformers hub.py file

This script patches the transformers hub.py file to fix the OfflineModeIsEnabled import error.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("patch-hub")

# Path to the hub.py file - assuming script is run from project root
# or from within the scripts directory
if os.path.basename(os.getcwd()) == "scripts":
    # If run from scripts directory
    HUB_PATH = "../.venv/lib/python3.12/site-packages/transformers/utils/hub.py"
else:
    # If run from project root
    HUB_PATH = ".venv/lib/python3.12/site-packages/transformers/utils/hub.py"

def apply_patches():
    """Apply patches to fix the OfflineModeIsEnabled import error."""
    logger.info(f"Patching {HUB_PATH}")
    
    # Create backup if it doesn't exist
    if not os.path.exists(HUB_PATH + ".bak"):
        with open(HUB_PATH, "r") as f:
            content = f.read()
        
        with open(HUB_PATH + ".bak", "w") as f:
            f.write(content)
        logger.info(f"Created backup at {HUB_PATH}.bak")
    
    # Read the hub.py file
    with open(HUB_PATH, "r") as f:
        lines = f.readlines()
    
    # Find the import section
    import_section_idx = None
    for i, line in enumerate(lines):
        if "from huggingface_hub.utils import (" in line:
            import_section_idx = i
            break
    
    if import_section_idx is None:
        logger.error("Could not find the import section")
        return False
    
    # Check if OfflineModeIsEnabled is already in the imports
    offline_in_imports = False
    closing_parenthesis_idx = None
    
    for i in range(import_section_idx + 1, len(lines)):
        if "OfflineModeIsEnabled," in lines[i]:
            offline_in_imports = True
        if ")" in lines[i]:
            closing_parenthesis_idx = i
            break
    
    # If OfflineModeIsEnabled is in imports but doesn't exist, we need to patch it
    if offline_in_imports:
        # Add our own implementation before the imports
        patch_code = """# Patch for OfflineModeIsEnabled
class OfflineModeIsEnabled(Exception):
    \"\"\"Custom implementation for compatibility.\"\"\"
    pass

"""
        
        # Insert the patch before the imports
        lines.insert(import_section_idx, patch_code)
        
        # Remove the import of OfflineModeIsEnabled
        for i in range(import_section_idx + 4, closing_parenthesis_idx + 4):  # +4 because we added 4 lines
            if "OfflineModeIsEnabled," in lines[i]:
                lines[i] = lines[i].replace("OfflineModeIsEnabled,", "")
                # If the line is now empty except for whitespace, remove it
                if lines[i].strip() == ",":
                    lines[i] = ""
                break
    
        # Write the patched file
        with open(HUB_PATH, "w") as f:
            f.writelines(lines)
        
        logger.info("Successfully patched hub.py")
        return True
    else:
        logger.info("No need to patch hub.py")
        return True

if __name__ == "__main__":
    apply_patches() 