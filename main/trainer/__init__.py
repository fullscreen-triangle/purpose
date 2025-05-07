"""
Trainer module for Purpose project.

This module contains components for training language models on processed data.
"""

# Import core trainer classes
from purpose.trainer.base import BaseTrainer, TrainerConfig, TrainingOutput
from purpose.trainer.transformers_trainer import TransformersTrainer
from purpose.trainer.knowledge_distillation import KnowledgeDistiller, run_distillation
from purpose.trainer.enhanced_distillation import EnhancedDistiller, run_enhanced_distillation

# Define the public API
__all__ = [
    'BaseTrainer',
    'TrainerConfig',
    'TrainingOutput',
    'TransformersTrainer',
    'KnowledgeDistiller',
    'run_distillation',
    'EnhancedDistiller',
    'run_enhanced_distillation'
]
