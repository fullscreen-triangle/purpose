"""
Base classes for inference in Purpose project.
"""

import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, NamedTuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main.log"),
    ]
)
logger = logging.getLogger("main")

@dataclass
class InferenceConfig:
    """Configuration class for inference engines."""
    model_dir: str
    device: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.7
    
    def __post_init__(self):
        """Convert string paths to Path objects after initialization."""
        self.model_dir = Path(self.model_dir)

class InferenceOutput(NamedTuple):
    """Output data from inference execution."""
    generated_text: str
    prompt: str
    model_name: str
    metadata: Dict[str, Any]

class BaseInference(ABC):
    """
    Base class for model inference in Purpose project.
    
    This class provides the abstract interface and common functionality
    for generating text using trained language models.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
    ):
        """
        Initialize the inference engine.
        
        Args:
            config: Configuration for the inference engine
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.logger = logger
        
        # Load model and tokenizer
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model and tokenizer.
        """
        pass
    
    @abstractmethod
    def generate_text(
        self, 
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text as a string
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        pass 