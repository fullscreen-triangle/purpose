"""
Model training module for domain-specific LLM training.
"""

from purpose.trainer.base import ModelTrainer
from purpose.trainer.transformers_trainer import TransformersTrainer

# Use TransformersTrainer as the default implementation
ModelTrainer = TransformersTrainer

__all__ = [
    "ModelTrainer",
    "TransformersTrainer"
]
