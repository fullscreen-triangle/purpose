---
layout: default
title: API Reference
nav_order: 4
---

# API Reference

This page provides detailed documentation for all public APIs in the Purpose package.

## Core Classes

### Processor

The main processing class for handling domain-specific operations.

```python
class Processor:
    def __init__(self, config=None):
        """
        Initialize a new Processor instance.
        
        Args:
            config (Config, optional): Configuration object
        """
        
    def process_data(self, data):
        """
        Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
```

### DistributedProcessor

Handles distributed processing across multiple workers.

```python
class DistributedProcessor:
    def __init__(self, workers=4):
        """
        Initialize distributed processor.
        
        Args:
            workers (int): Number of worker processes
        """
        
    def process_batch(self, data_batch):
        """
        Process a batch of data in parallel.
        
        Args:
            data_batch: Batch of data to process
            
        Returns:
            List of processed results
        """
```

### ModelOptimizer

Handles model optimization and performance tuning.

```python
class ModelOptimizer:
    def __init__(self, strategy='default'):
        """
        Initialize model optimizer.
        
        Args:
            strategy (str): Optimization strategy
        """
        
    def optimize(self, model):
        """
        Optimize the given model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
```

### Distiller

Handles knowledge distillation between teacher and student models.

```python
class Distiller:
    def __init__(self, teacher_model, student_model):
        """
        Initialize distiller.
        
        Args:
            teacher_model: The teacher model
            student_model: The student model
        """
        
    def distill(self, training_data):
        """
        Perform knowledge distillation.
        
        Args:
            training_data: Data for distillation
            
        Returns:
            Distilled student model
        """
```

## Utility Functions

### Configuration

```python
def load_config(path):
    """
    Load configuration from file.
    
    Args:
        path (str): Path to config file
        
    Returns:
        Config object
    """

def save_config(config, path):
    """
    Save configuration to file.
    
    Args:
        config (Config): Config object
        path (str): Output path
    """
```

### Data Processing

```python
def preprocess_data(data):
    """
    Preprocess input data.
    
    Args:
        data: Raw input data
        
    Returns:
        Preprocessed data
    """

def postprocess_results(results):
    """
    Postprocess results.
    
    Args:
        results: Raw results
        
    Returns:
        Processed results
    """
```

## Constants

```python
DEFAULT_BATCH_SIZE = 32
MAX_WORKERS = 8
OPTIMIZATION_LEVELS = ['low', 'medium', 'high']
```

## Exceptions

```python
class ProcessingError(Exception):
    """Raised when processing fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class OptimizationError(Exception):
    """Raised when optimization fails"""
    pass
```

For more detailed examples and use cases, please refer to the [Examples](examples.md) section. 