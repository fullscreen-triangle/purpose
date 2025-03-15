"""
Base classes for model training in Domain-LLM.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# Configure logging if not already configured
logger = logging.getLogger("domain-llm")

class ModelTrainer(ABC):
    """
    Base class for training language models on domain-specific data.
    
    This class provides the abstract interface and common functionality
    for training language models on processed domain data.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "models",
        model_name: str = "gpt2",
        use_lora: bool = False,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing the processed data
            output_dir: Directory to save the trained model
            model_name: Name or path of the base model to fine-tune
            use_lora: Whether to use parameter-efficient fine-tuning with LoRA
            device: Device to use for training (None for auto-detection)
            **kwargs: Additional trainer-specific arguments
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store additional configuration
        self.config = kwargs
        self.logger = logger
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def train(
        self,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        **kwargs
    ) -> None:
        """
        Train the model on the processed data.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate scheduler
            **kwargs: Additional training arguments
        """
        pass
    
    @abstractmethod
    def load_data(self, corpus_file: str = "purpose_corpus.txt"):
        """
        Load the training data from the processed corpus.
        
        Args:
            corpus_file: Name of the corpus file in the data directory
        """
        pass
    
    @abstractmethod
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model (default: output_dir/model_name)
            
        Returns:
            Path where the model was saved
        """
        pass
    
    def evaluate(self, test_data = None, **kwargs):
        """
        Evaluate the trained model.
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation metrics
        """
        # Default implementation returns None
        # This can be overridden by subclasses
        return None 