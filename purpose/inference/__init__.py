"""
Model inference module for using domain-specific LLMs.
"""

from purpose.inference.model import ModelInference
from purpose.inference.interface import InteractiveInterface, QAInterface

__all__ = [
    "ModelInference",
    "InteractiveInterface",
    "QAInterface"
]
