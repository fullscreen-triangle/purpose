"""
Model training module for domain-specific LLM training.
"""

from domain_llm.trainer.base import ModelTrainer
from domain_llm.trainer.transformers_trainer import TransformersTrainer

# Use TransformersTrainer as the default implementation
ModelTrainer = TransformersTrainer

__all__ = [
    "ModelTrainer",
    "TransformersTrainer"
]
