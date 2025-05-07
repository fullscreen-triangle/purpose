import os
import logging
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

def load_llama_model(
    model_path: str,
    bit_precision: int = 4,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    **kwargs
) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
    """
    Load a LLaMA model and tokenizer with the specified configuration.
    
    Args:
        model_path: Path to the LLaMA model or model identifier
        bit_precision: Bit precision for quantization (4 or 8)
        device_map: Device mapping strategy
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        **kwargs: Additional arguments to pass to the model loading function
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading LLaMA model from {model_path}")
    
    # Configure quantization settings based on bit precision
    if bit_precision not in [4, 8, 16, 32]:
        logger.warning(f"Invalid bit precision {bit_precision}, defaulting to 4-bit")
        bit_precision = 4
    
    # Determine quantization settings
    quantization_config = None
    if bit_precision == 4 or load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif bit_precision == 8 or load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        logger.info(f"Loading model in {bit_precision}-bit precision")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise
    
    # Load model with appropriate configuration
    try:
        model_kwargs = {
            "device_map": device_map,
            "quantization_config": quantization_config if quantization_config else None,
            "torch_dtype": torch.float16 if bit_precision < 16 else torch.float32,
            **kwargs
        }
        
        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        logger.info(f"Model loaded successfully with device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else device_map}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_model_info(model: LlamaForCausalLM) -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Args:
        model: The loaded LLaMA model
        
    Returns:
        Dictionary with model information
    """
    info = {
        "model_type": model.config.model_type,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device_map": model.hf_device_map if hasattr(model, "hf_device_map") else "N/A",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "memory_usage": {
            "gpu": {i: f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB" 
                  for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else "N/A"
        }
    }
    
    return info 