"""
Trainer module for Purpose project.

This module contains components for training language models on processed data.
"""

# Import core trainer classes
from main.trainer.base import BaseTrainer
from main.trainer.transformers_trainer import TransformersTrainer
from main.trainer.knowledge_distillation import KnowledgeDistiller, run_distillation
from main.trainer.enhanced_distillation import EnhancedDistiller, run_enhanced_distillation

# Define the public API
__all__ = [
    'BaseTrainer',
    'TransformersTrainer',
    'KnowledgeDistiller',
    'run_distillation',
    'EnhancedDistiller',
    'run_enhanced_distillation'
]
