"""
Model inference module for using domain-specific LLMs.
"""

from domain_llm.inference.model import ModelInference
from domain_llm.inference.interface import InteractiveInterface, QAInterface

__all__ = [
    "ModelInference",
    "InteractiveInterface",
    "QAInterface"
]
