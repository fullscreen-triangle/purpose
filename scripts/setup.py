#!/usr/bin/env python3
"""
Purpose Project Setup Script

This script sets up the Purpose project environment by:
1. Installing dependencies from requirements.txt
2. Applying necessary patches for compatibility
3. Creating required directories
4. Validating the environment

For detailed project documentation, see the README.md file.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("purpose-setup")

def get_project_root():
    """Get the absolute path to the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "scripts":
        return os.path.dirname(script_dir)
    return script_dir

def install_requirements(requirements_file="requirements.txt"):
    """Install project dependencies from the specified requirements file."""
    project_root = get_project_root()
    req_path = os.path.join(project_root, requirements_file)
    
    logger.info(f"Installing dependencies from {req_path}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", req_path
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def apply_patches():
    """Apply necessary patches for compatibility."""
    logger.info("Applying compatibility patches")
    try:
        # Import and apply the transformers patch
        sys.path.insert(0, get_project_root())
        try:
            from scripts.patch_hub import apply_patches as apply_transform_patches
            success = apply_transform_patches()
            if success:
                logger.info("Transformers compatibility patch applied successfully")
            else:
                logger.warning("No need to apply transformers patch")
            return success
        except ImportError:
            logger.error("Failed to import patch module - ensure patch_hub.py exists in scripts directory")
            return False
    except Exception as e:
        logger.error(f"Failed to apply patches: {e}")
        return False

def create_directories():
    """Create necessary project directories if they don't exist."""
    project_root = get_project_root()
    
    # Directories to ensure exist
    directories = [
        "data/processed",
        "models",
        "content/papers",
        "logs"
    ]
    
    logger.info("Creating necessary project directories")
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def validate_environment():
    """Validate the Python environment and dependencies."""
    logger.info("Validating environment")
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        logger.warning(f"Python {py_version.major}.{py_version.minor} detected. Python 3.8+ recommended.")
    else:
        logger.info(f"Python version {py_version.major}.{py_version.minor} - OK")
    
    # Check key dependencies
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not installed - required for model processing")
    
    try:
        import huggingface_hub
        logger.info(f"Huggingface Hub version: {huggingface_hub.__version__}")
    except ImportError:
        logger.error("Huggingface Hub not installed - required for model access")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        logger.error("PyTorch not installed - required for model training")

def setup_virtual_environment():
    """Set up a virtual environment for the project."""
    project_root = get_project_root()
    venv_dir = os.path.join(project_root, ".venv")
    
    if os.path.exists(venv_dir):
        logger.info(f"Virtual environment already exists at: {venv_dir}")
        return True
    
    logger.info(f"Creating virtual environment at: {venv_dir}")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        logger.info("Virtual environment created successfully")
        
        # Provide activation instructions
        logger.info("\nTo activate the virtual environment:")
        if os.name == "nt":  # Windows
            logger.info(f"   .venv\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            logger.info(f"   source .venv/bin/activate")
        logger.info("\nThen run this setup script again to install dependencies")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up the Purpose project environment")
    parser.add_argument("--venv", action="store_true", help="Create virtual environment only")
    parser.add_argument("--requirements", action="store_true", help="Install requirements only")
    parser.add_argument("--patch", action="store_true", help="Apply patches only")
    parser.add_argument("--dirs", action="store_true", help="Create directories only")
    args = parser.parse_args()
    
    # If specific actions are requested, only perform those
    if any([args.venv, args.requirements, args.patch, args.dirs]):
        if args.venv:
            setup_virtual_environment()
        if args.requirements:
            install_requirements()
        if args.patch:
            apply_patches()
        if args.dirs:
            create_directories()
        return
    
    # Otherwise perform full setup
    logger.info("Starting Purpose project setup")
    
    # Display project info
    project_root = get_project_root()
    logger.info(f"Project root directory: {project_root}")
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    logger.info(f"Running in virtual environment: {in_venv}")
    
    if not in_venv:
        setup_virtual_environment()
        logger.info("Please activate the virtual environment and run setup again")
        return
    
    # Full setup sequence
    if install_requirements():
        create_directories()
        apply_patches()
        validate_environment()
        
        logger.info("\n✓ Purpose project setup complete!")
        logger.info("\nNext steps:")
        logger.info("1. Place your domain papers in the content/papers directory")
        logger.info("2. Run: python scripts/run_distillation.py --papers-dir content/papers --model-name distilgpt2")
        logger.info("   Or use the CLI: purpose enhanced-distill --papers-dir content/papers")
    else:
        logger.error("Setup failed - see error messages above")

if __name__ == "__main__":
    main() 