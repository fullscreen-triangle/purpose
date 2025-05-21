"""
Pipelines module for the Purpose project.

This module provides pipeline functionality for chaining operations across the project.
"""

from main.pipelines.base import BasePipeline, ConditionalPipeline, FunctionStage

__all__ = [
    'BasePipeline',
    'ConditionalPipeline',
    'FunctionStage'
] 