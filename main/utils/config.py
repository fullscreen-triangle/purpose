"""
Configuration system for the Purpose project.

This module provides a unified configuration system that can be used across all components.
"""

import os
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, cast
from dataclasses import dataclass, asdict, is_dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("purpose.log"),
    ]
)
logger = logging.getLogger("purpose")

# Type variables for generic type hints
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base configuration class that all config classes should inherit from."""
    
    # Default configurations directory
    CONFIG_DIR: str = field(default="configs", metadata={"description": "Directory containing configuration files"})
    
    @classmethod
    def from_dict(cls: Type[ConfigType], config_dict: Dict[str, Any]) -> ConfigType:
        """
        Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Configuration object
        """
        # Filter out keys that are not in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls: Type[ConfigType], file_path: Union[str, Path]) -> ConfigType:
        """
        Load configuration from a file (JSON or YAML).
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Configuration object
            
        Raises:
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load the file based on extension
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls: Type[ConfigType], prefix: str = "PURPOSE") -> ConfigType:
        """
        Load configuration from environment variables.
        
        Environment variables are expected to follow the pattern:
        PREFIX_KEY=value (e.g., PURPOSE_MODEL_NAME=gpt2)
        
        Args:
            prefix: Prefix for environment variables
            
        Returns:
            Configuration object
        """
        # Get all environment variables with the given prefix
        env_vars = {
            k[len(prefix) + 1:].lower(): v
            for k, v in os.environ.items()
            if k.startswith(f"{prefix}_")
        }
        
        return cls.from_dict(env_vars)
    
    @classmethod
    def from_args(cls: Type[ConfigType], args: Optional[argparse.Namespace] = None) -> ConfigType:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Configuration object
        """
        if args is None:
            # Create default parser with fields based on dataclass
            parser = argparse.ArgumentParser()
            for field_name, field_info in cls.__dataclass_fields__.items():
                # Skip private fields
                if field_name.startswith('_'):
                    continue
                
                # Get description from metadata if available
                description = field_info.metadata.get('description', f"{field_name} configuration value")
                
                # Add argument with appropriate type
                parser.add_argument(
                    f"--{field_name.replace('_', '-')}",
                    dest=field_name,
                    help=description,
                    type=field_info.type
                )
            
            # Parse arguments
            args = parser.parse_args()
        
        # Convert args to dictionary and filter None values
        args_dict = {
            k: v for k, v in vars(args).items()
            if v is not None and k in cls.__dataclass_fields__
        }
        
        return cls.from_dict(args_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the configuration to a file (JSON or YAML).
        
        Args:
            file_path: Path to save the configuration to
            
        Raises:
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file based on extension
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def merge(self, other: Union[Dict[str, Any], 'BaseConfig']) -> None:
        """
        Merge another configuration or dictionary into this one.
        
        Args:
            other: Another configuration or dictionary to merge
        """
        if isinstance(other, dict):
            other_dict = other
        elif is_dataclass(other):
            other_dict = asdict(other)
        else:
            raise TypeError(f"Cannot merge with object of type {type(other)}")
        
        # Update fields that exist in this config
        for field_name in self.__dataclass_fields__:
            if field_name in other_dict and other_dict[field_name] is not None:
                setattr(self, field_name, other_dict[field_name])

@dataclass
class ProjectConfig(BaseConfig):
    """
    Main configuration for the Purpose project.
    
    This configuration contains settings that apply to the entire project.
    """
    
    # Project settings
    project_name: str = field(default="purpose", metadata={"description": "Name of the project"})
    project_version: str = field(default="0.1.0", metadata={"description": "Version of the project"})
    
    # Directory settings
    data_dir: str = field(default="data", metadata={"description": "Directory for data files"})
    models_dir: str = field(default="models", metadata={"description": "Directory for model files"})
    output_dir: str = field(default="output", metadata={"description": "Directory for output files"})
    
    # Logging settings
    log_level: str = field(default="INFO", metadata={"description": "Logging level"})
    log_file: str = field(default="purpose.log", metadata={"description": "Log file path"})
    
    # Hardware settings
    use_gpu: bool = field(default=True, metadata={"description": "Whether to use GPU if available"})
    gpu_device: Optional[int] = field(default=None, metadata={"description": "GPU device ID to use"})
    use_distributed: bool = field(default=False, metadata={"description": "Whether to use distributed processing"})
    
    def __post_init__(self):
        """Initialize paths as Path objects."""
        self.data_dir = str(Path(self.data_dir))
        self.models_dir = str(Path(self.models_dir))
        self.output_dir = str(Path(self.output_dir))
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.output_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set up logging level
        logging.getLogger("purpose").setLevel(getattr(logging, self.log_level))

@dataclass
class ProcessorConfig(BaseConfig):
    """
    Configuration for data processors.
    
    This configuration contains settings specific to data processing.
    """
    
    data_dir: str = field(default="data", metadata={"description": "Directory containing raw data files"})
    output_dir: str = field(default="data/processed", metadata={"description": "Directory to save processed data"})
    output_corpus_filename: str = field(default="purpose_corpus.txt", metadata={"description": "Filename for the output text corpus"})
    output_jsonl_filename: str = field(default="domain_data.jsonl", metadata={"description": "Filename for the output JSONL file"})
    max_file_size_mb: float = field(default=50.0, metadata={"description": "Maximum file size to process in MB"})
    
    def __post_init__(self):
        """Convert string paths to Path objects after initialization."""
        self.data_dir = str(Path(self.data_dir))
        self.output_dir = str(Path(self.output_dir))
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class TrainerConfig(BaseConfig):
    """
    Configuration for model trainers.
    
    This configuration contains settings specific to model training.
    """
    
    data_dir: str = field(default="data/processed", metadata={"description": "Directory containing processed data"})
    output_dir: str = field(default="models", metadata={"description": "Directory to save trained models"})
    model_name: str = field(default="gpt2", metadata={"description": "Name or path of the base model"})
    model_output_name: str = field(default="purpose_model", metadata={"description": "Name for the output model"})
    
    # Training settings
    batch_size: int = field(default=4, metadata={"description": "Training batch size"})
    learning_rate: float = field(default=5e-5, metadata={"description": "Learning rate"})
    num_epochs: int = field(default=3, metadata={"description": "Number of training epochs"})
    warmup_steps: int = field(default=100, metadata={"description": "Warmup steps for learning rate scheduler"})
    
    # Model settings
    use_lora: bool = field(default=False, metadata={"description": "Whether to use LoRA for parameter-efficient fine-tuning"})
    lora_r: int = field(default=8, metadata={"description": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"description": "LoRA alpha parameter"})
    
    # Hardware settings
    device: Optional[str] = field(default=None, metadata={"description": "Device to use for training (None for auto-detection)"})
    fp16: bool = field(default=False, metadata={"description": "Whether to use mixed precision training"})
    
    def __post_init__(self):
        """Convert string paths to Path objects after initialization."""
        self.data_dir = str(Path(self.data_dir))
        self.output_dir = str(Path(self.output_dir))
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class InferenceConfig(BaseConfig):
    """
    Configuration for model inference.
    
    This configuration contains settings specific to model inference.
    """
    
    model_dir: str = field(default="models/purpose_model", metadata={"description": "Directory containing the trained model"})
    
    # Generation settings
    max_length: int = field(default=512, metadata={"description": "Maximum length of generated text"})
    temperature: float = field(default=0.7, metadata={"description": "Temperature for text generation"})
    top_p: float = field(default=0.9, metadata={"description": "Top-p sampling parameter"})
    top_k: int = field(default=50, metadata={"description": "Top-k sampling parameter"})
    repetition_penalty: float = field(default=1.1, metadata={"description": "Repetition penalty"})
    
    # Hardware settings
    device: Optional[str] = field(default=None, metadata={"description": "Device to use for inference (None for auto-detection)"})
    
    def __post_init__(self):
        """Convert string paths to Path objects after initialization."""
        self.model_dir = str(Path(self.model_dir))

# Create a global configuration instance
config = ProjectConfig()

def load_config(config_path: Optional[Union[str, Path]] = None) -> ProjectConfig:
    """
    Load configuration from a file and environment variables.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        Loaded configuration
    """
    global config
    
    # Load from file if provided
    if config_path:
        try:
            file_config = ProjectConfig.from_file(config_path)
            config.merge(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {str(e)}")
    
    # Load from environment variables
    try:
        env_config = ProjectConfig.from_env()
        config.merge(env_config)
    except Exception as e:
        logger.warning(f"Error loading configuration from environment variables: {str(e)}")
    
    return config

def get_config() -> ProjectConfig:
    """
    Get the global configuration.
    
    Returns:
        Global configuration
    """
    return config

def get_processor_config() -> ProcessorConfig:
    """
    Get the processor configuration.
    
    Returns:
        Processor configuration
    """
    processor_config = ProcessorConfig()
    
    # Copy relevant fields from the global config
    processor_config.data_dir = config.data_dir
    processor_config.output_dir = os.path.join(config.data_dir, "processed")
    
    return processor_config

def get_trainer_config() -> TrainerConfig:
    """
    Get the trainer configuration.
    
    Returns:
        Trainer configuration
    """
    trainer_config = TrainerConfig()
    
    # Copy relevant fields from the global config
    trainer_config.data_dir = os.path.join(config.data_dir, "processed")
    trainer_config.output_dir = config.models_dir
    
    # Set device based on global config
    if config.use_gpu and not trainer_config.device:
        trainer_config.device = f"cuda:{config.gpu_device}" if config.gpu_device is not None else "cuda"
    
    return trainer_config

def get_inference_config() -> InferenceConfig:
    """
    Get the inference configuration.
    
    Returns:
        Inference configuration
    """
    inference_config = InferenceConfig()
    
    # Copy relevant fields from the global config
    inference_config.model_dir = os.path.join(config.models_dir, "purpose_model")
    
    # Set device based on global config
    if config.use_gpu and not inference_config.device:
        inference_config.device = f"cuda:{config.gpu_device}" if config.gpu_device is not None else "cuda"
    
    return inference_config 