"""
Service registration for the Purpose project.

This module registers all the services with the dependency injection container.
"""
from main.inference.base import BaseInference
from main.inference.model import ModelInference
from main.processor.base import BaseProcessor, CombinedDataProcessor
from main.processor.text_processor import SprintTextProcessor
from main.trainer.base import BaseTrainer
from main.trainer.transformers_trainer import TransformersTrainer
from main.utils.di import global_provider


def register_services():
    """
    Register all services with the global service provider.
    """
    # Register processors
    global_provider.register('base_processor', BaseProcessor, singleton=False)
    global_provider.register('text_processor', SprintTextProcessor, singleton=False)
    global_provider.register('combined_processor', CombinedDataProcessor, singleton=False)
    
    # Register trainers
    global_provider.register('base_trainer', BaseTrainer, singleton=False)
    global_provider.register('transformers_trainer', TransformersTrainer, singleton=False)
    
    # Register inference engines
    global_provider.register('base_inference', BaseInference, singleton=False)
    global_provider.register('model_inference', ModelInference, singleton=False)
    
    # Register factory methods for common configurations
    global_provider.register('default_processor', lambda provider: provider.resolve('combined_processor'), singleton=True)
    global_provider.register('default_trainer', lambda provider: provider.resolve('transformers_trainer'), singleton=True)
    global_provider.register('default_inference', lambda provider: provider.resolve('model_inference'), singleton=True)

# Register services on module import
register_services()

def get_processor(processor_type: str = 'default_processor'):
    """
    Get a processor instance.
    
    Args:
        processor_type: The type of processor to get
        
    Returns:
        The processor instance
    """
    return global_provider.resolve(processor_type)

def get_trainer(trainer_type: str = 'default_trainer'):
    """
    Get a trainer instance.
    
    Args:
        trainer_type: The type of trainer to get
        
    Returns:
        The trainer instance
    """
    return global_provider.resolve(trainer_type)

def get_inference(inference_type: str = 'default_inference'):
    """
    Get an inference engine instance.
    
    Args:
        inference_type: The type of inference engine to get
        
    Returns:
        The inference engine instance
    """
    return global_provider.resolve(inference_type) 