"""
Inference module for Purpose project.

This module contains components for using trained language models for inference.
"""

# Import core inference classes
from purpose.inference.base import BaseInference, InferenceConfig, InferenceOutput
from purpose.inference.model import ModelInference
from purpose.inference.interface import InteractiveInterface, QAInterface

# Define the public API
__all__ = [
    'BaseInference',
    'InferenceConfig',
    'InferenceOutput',
    'ModelInference',
    'InteractiveInterface',
    'QAInterface'
]
