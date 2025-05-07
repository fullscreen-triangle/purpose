"""
Pipelines module for the Purpose project.

This module provides pipeline functionality for chaining operations across the project.
"""

from purpose.pipelines.base import BasePipeline, ConditionalPipeline, FunctionStage

__all__ = [
    'BasePipeline',
    'ConditionalPipeline',
    'FunctionStage'
] 