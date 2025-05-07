"""
Processor module for Purpose project.

This module contains components for processing raw data into suitable formats for training language models.
"""

# Import core processor classes
from purpose.processor.base import BaseProcessor, ProcessorConfig, ProcessorOutput
from purpose.processor.text_processor import TextProcessor
from purpose.processor.budget_processor import BudgetDataProcessor
from purpose.processor.codespace_processor import CodespaceProcessor

# Import the combined processor (main entry point)
from purpose.processor.base import CombinedDataProcessor

# Define the public API
__all__ = [
    'BaseProcessor',
    'ProcessorConfig',
    'ProcessorOutput',
    'TextProcessor',
    'BudgetDataProcessor',
    'CodespaceProcessor',
    'CombinedDataProcessor',
]
