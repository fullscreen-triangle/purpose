import logging
from typing import Dict, Any, List, Optional, Union

import torch
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    StoppingCriteriaList,
    StoppingCriteria
)

from ..utils.llama_utils import load_llama_model

logger = logging.getLogger(__name__)

class LlamaInference:
    """
    Handler for local inference using LLaMA models.
    """
    
    def __init__(
        self,
        model_path: str,
        bit_precision: int = 4,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **model_kwargs
    ):
        """
        Initialize the LLaMA inference handler.
        
        Args:
            model_path: Path to the LLaMA model
            bit_precision: Bit precision for model quantization
            device_map: Device mapping strategy
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            **model_kwargs: Additional arguments for model loading
        """
        self.model_path = model_path
        self.bit_precision = bit_precision
        self.device_map = device_map
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Load model and tokenizer
        logger.info(f"Initializing LLaMA inference with model: {model_path}")
        self.model, self.tokenizer = load_llama_model(
            model_path=model_path,
            bit_precision=bit_precision,
            device_map=device_map,
            **model_kwargs
        )
        
        # Set default tokenizer padding settings
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        logger.info("LLaMA inference initialized successfully")
    
    def format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format the prompt with an optional system prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt for instruction
            
        Returns:
            Formatted prompt string
        """
        if system_prompt:
            # Use a LLaMA-specific prompt format with system instructions
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            # Basic instruction format
            return f"<s>[INST] {prompt} [/INST]"
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system instructions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for token repetition
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Use provided parameters or fall back to instance defaults
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        repetition_penalty = repetition_penalty or self.repetition_penalty
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate output
        with torch.no_grad():
            generation_output = self.model.generate(**gen_kwargs)
            
        # Decode the generated tokens, skip input prompt
        input_length = input_ids.shape[1]
        generated_tokens = generation_output[0, input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Create response
        result = {
            "generated_text": generated_text,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tokens_generated": len(generated_tokens),
            "generation_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty
            }
        }
        
        return result
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of input prompts
            system_prompt: Optional system instructions to use for all prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            List of dictionaries containing the generated texts and metadata
        """
        return [self.generate(prompt, system_prompt, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        from ..utils.llama_utils import get_model_info
        return get_model_info(self.model) 