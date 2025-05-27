"""
Processor module for Purpose project.

This module contains components for processing raw data into suitable formats for training language models.
"""

# Import core processor classes
from main.processor.base import BaseProcessor, ProcessorConfig, ProcessorOutput
from main.processor.text_processor import SprintTextProcessor
from main.processor.budget_processor import BudgetProcessor
from main.processor.codespace_processor import CodespaceProcessor

# Import the combined processor (main entry point)
from main.processor.base import CombinedDataProcessor

# Define the public API
__all__ = [
    'BaseProcessor',
    'ProcessorConfig',
    'ProcessorOutput',
    'SprintTextProcessor',
    'BudgetProcessor',
    'CodespaceProcessor',
    'CombinedDataProcessor',
]
