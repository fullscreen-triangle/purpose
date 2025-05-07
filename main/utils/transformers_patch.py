"""
Transformers Patch Module

This module patches compatibility issues between different versions of transformers
and huggingface_hub libraries.
"""

import sys
import importlib
from types import ModuleType
import logging

logger = logging.getLogger(__name__)

class HuggingfaceHubUtilsPatch(ModuleType):
    """Patch module for huggingface_hub.utils to provide compatibility classes."""
    
    def __init__(self):
        super().__init__("huggingface_hub.utils")
        
        # Get the original module
        self.original_module = importlib.import_module("huggingface_hub.utils")
        
        # Copy all attributes from the original module
        for attr in dir(self.original_module):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(self.original_module, attr))
    
    @property
    def OfflineModeIsEnabled(self):
        """
        Provide a compatibility class for OfflineModeIsEnabled.
        In newer versions, this might be HfHubHTTPError or another exception.
        """
        # Try to use the existing offline exception if it exists
        if hasattr(self.original_module, "OfflineModeIsEnabled"):
            return getattr(self.original_module, "OfflineModeIsEnabled")
        elif hasattr(self.original_module, "HfHubHTTPError"):
            return getattr(self.original_module, "HfHubHTTPError")
        else:
            # If neither exists, create a custom exception class
            class OfflineModeIsEnabled(Exception):
                """Compatibility class for offline mode exceptions."""
                pass
            return OfflineModeIsEnabled


def monkey_patch_transformers():
    """Monkey patch transformers hub.py to avoid the import error."""
    try:
        import transformers.utils.hub
        
        # Check if the problematic import is present
        original_code = transformers.utils.hub.__file__
        
        # Create a new version of the imports
        def patch_imports():
            if not hasattr(transformers.utils.hub, "OfflineModeIsEnabled"):
                # Define the class if it doesn't exist
                class OfflineModeIsEnabled(Exception):
                    """Compatibility class for offline mode exceptions."""
                    pass
                
                # Add it to the transformers.utils.hub module
                transformers.utils.hub.OfflineModeIsEnabled = OfflineModeIsEnabled
                
                logger.info("Patched OfflineModeIsEnabled in transformers.utils.hub")
        
        # Apply the patch
        patch_imports()
        
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to monkey patch transformers hub: {e}")


def apply_patches():
    """Apply all necessary patches to fix compatibility issues."""
    # Patch huggingface_hub.utils
    try:
        sys.modules["huggingface_hub.utils"] = HuggingfaceHubUtilsPatch()
        logger.info("Applied huggingface_hub.utils compatibility patch")
    except Exception as e:
        logger.warning(f"Failed to patch huggingface_hub.utils: {e}")
    
    # Monkey patch transformers.utils.hub
    monkey_patch_transformers()
    
    logger.info("Applied transformers compatibility patches")


# Only apply patches when explicitly called
if __name__ == "__main__":
    apply_patches() 