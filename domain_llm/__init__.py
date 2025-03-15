"""
Domain-LLM - A framework for creating domain-specific language models
"""

__version__ = "0.1.0"
__author__ = "Sprint Team"

from domain_llm.processor import DataProcessor
from domain_llm.trainer import ModelTrainer
from domain_llm.inference import ModelInference

__all__ = ["DataProcessor", "ModelTrainer", "ModelInference"]
