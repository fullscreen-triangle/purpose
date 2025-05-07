"""
Interfaces for the Purpose project.

This module defines the contracts between different modules in the project.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Protocol
from pathlib import Path
from dataclasses import dataclass

# Processor Interfaces
class ProcessorInterface(Protocol):
    """Protocol defining the interface for data processors."""
    
    def process(self) -> Any:
        """
        Process data according to the processor's configuration.
        
        Returns:
            Processed data output
        """
        ...

# Trainer Interfaces
class TrainerInterface(Protocol):
    """Protocol defining the interface for model trainers."""
    
    def load_data(self, corpus_file: str = "purpose_corpus.txt") -> None:
        """
        Load and prepare data for training.
        
        Args:
            corpus_file: Name of the corpus file to load
        """
        ...
    
    def train(self, **kwargs) -> Any:
        """
        Train the model on the prepared data.
        
        Args:
            **kwargs: Training parameters
            
        Returns:
            Training output
        """
        ...
    
    def save_model(self, path: Optional[str] = None) -> Path:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path to the saved model
        """
        ...

# Inference Interfaces
class InferenceInterface(Protocol):
    """Protocol defining the interface for model inference."""
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        ...

# Data Exchange Interfaces
@dataclass
class DataFormat:
    """Base class for data exchange formats between components."""
    version: str = "1.0"

@dataclass
class TextRecord(DataFormat):
    """Text record for exchange between components."""
    text: str
    metadata: Dict[str, Any] = None

@dataclass
class TrainingRecord(DataFormat):
    """Training record for exchange between components."""
    input_text: str
    target_text: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ModelPrediction(DataFormat):
    """Model prediction for exchange between components."""
    input_text: str
    output_text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

# Pipeline Interfaces
class PipelineStage(Protocol):
    """Protocol defining a pipeline stage interface."""
    
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return output data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        ...

class Pipeline(Protocol):
    """Protocol defining a pipeline interface."""
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            stage: Pipeline stage to add
        """
        ...
    
    def run(self, input_data: Any) -> Any:
        """
        Run the pipeline on input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Pipeline output
        """
        ... 