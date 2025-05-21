"""
Inference module for Purpose project.

This module contains components for using trained language models for inference.
"""

# Import core inference classes
from main.inference.base import BaseInference, InferenceConfig, InferenceOutput
from main.inference.model import ModelInference
from main.inference.interface import InteractiveInterface, QAInterface

# Define the public API
__all__ = [
    'BaseInference',
    'InferenceConfig',
    'InferenceOutput',
    'ModelInference',
    'InteractiveInterface',
    'QAInterface'
]
