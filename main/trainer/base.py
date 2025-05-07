"""
Base Trainer Module

This module defines abstract base classes for model trainers in the Purpose framework.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union


class BaseTrainer(ABC):
    """
    Abstract base class for all model trainers in the Purpose framework.
    
    This class defines the common interface for trainers that can be used
    to train domain-specific language models.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path] = "models",
        model_name: str = "gpt2",
        use_lora: bool = False,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base trainer.
        
        Args:
            data_dir: Directory containing the processed data
            output_dir: Directory to save the trained model
            model_name: Name or path of the base model to fine-tune
            use_lora: Whether to use parameter-efficient fine-tuning with LoRA
            device: Device to use for training (None for auto-detection)
            **kwargs: Additional trainer-specific arguments
        """
        # Convert string paths to Path objects
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Determine device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Additional parameters
        self.kwargs = kwargs
        
        self.logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
        self.logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def load_data(self, corpus_file: str = "purpose_corpus.txt"):
        """
        Load the training data from the processed corpus.
        
        Args:
            corpus_file: Name of the corpus file in the data directory
            
        Returns:
            Dataset with the loaded data
        """
        pass
    
    @abstractmethod
    def train(
        self,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        **kwargs
    ) -> None:
        """
        Train the model on the processed data.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            **kwargs: Additional training arguments
        """
        pass
    
    @abstractmethod
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data=None, **kwargs):
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation metrics
        """
        pass


class ModelTrainer(BaseTrainer):
    """
    Compatibility class to ensure existing code works with the new base class structure.
    This class can be removed once all trainers are updated to use BaseTrainer.
    """
    pass 