#!/usr/bin/env python3
"""
Environment Setup Script for Domain Purpose

This script sets up the Python environment with compatible versions 
of transformers, huggingface-hub, and tokenizers for Python 3.12.
It also applies the necessary patches to fix compatibility issues.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("setup-env")

def install_requirements():
    """Install the fixed requirements file with compatible package versions."""
    logger.info("Installing required packages with compatible versions...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Package installation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False

def apply_transformers_patch():
    """Apply the transformers patch to fix compatibility issues."""
    logger.info("Applying transformers patch...")
    try:
        # Import and apply the patch
        sys.path.insert(0, os.path.abspath("scripts"))
        from scripts.patch_hub import apply_patches
        apply_patches()
        logger.info("Transformers patch applied successfully.")
        return True
    except ImportError:
        logger.error("Failed to import patch_hub module. Make sure the file exists in the scripts directory.")
        return False
    except Exception as e:
        logger.error(f"Failed to apply transformers patch: {e}")
        return False

def check_installation():
    """Check if the installed packages are working correctly."""
    logger.info("Checking installation...")
    try:
        import transformers
        import huggingface_hub
        import tokenizers
        
        logger.info(f"transformers version: {transformers.__version__}")
        logger.info(f"huggingface_hub version: {huggingface_hub.__version__}")
        logger.info(f"tokenizers version: {tokenizers.__version__}")
        
        # Test importing the previously problematic module
        from transformers.utils.hub import HfApi
        logger.info("Successfully imported transformers.utils.hub.HfApi")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def main():
    """Main function to set up the environment."""
    logger.info("Starting environment setup...")
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install required packages. Exiting.")
        return False
    
    # Apply transformers patch
    if not apply_transformers_patch():
        logger.error("Failed to apply transformers patch. Exiting.")
        return False
    
    # Check installation
    if not check_installation():
        logger.error("Installation check failed. You may need to manually fix the issues.")
        return False
    
    logger.info("Environment setup completed successfully!")
    logger.info("You can now run the distillation CLI without compatibility issues.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 