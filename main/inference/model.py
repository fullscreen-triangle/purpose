"""
Model loading and inference for domain-specific LLMs.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# Configure logging if not already configured
logger = logging.getLogger("domain-llm")

class ModelInference:
    """
    Class for loading and using trained domain-specific language models.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the model inference.
        
        Args:
            model_dir: Directory containing the trained model
            device: Device to use for inference (None for auto-detection)
            **kwargs: Additional inference-specific arguments
        """
        self.model_dir = Path(model_dir)
        
        # Store additional configuration
        self.config = kwargs
        self.logger = logger
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer to None
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained language model.
        """
        self.logger.info(f"Loading model from {self.model_dir}")
        
        # Check if model exists
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir))
            
            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt using the trained model.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        # Create text generation pipeline if not exists
        if self.generator is None:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,  # Use GPU if available
            )
        
        # Generate text
        result = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        if num_return_sequences == 1:
            return result[0]["generated_text"]
        else:
            return [item["generated_text"] for item in result]
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Answer a question using the domain-specific language model.
        
        Args:
            question: Question to answer
            context: Additional context for the question
            **kwargs: Additional generation parameters
            
        Returns:
            Generated answer
        """
        # Format prompt with question
        if context:
            prompt = f"""
The following is domain-specific information:
{context}

Question: {question}

Answer:
"""
        else:
            prompt = f"""
Question: {question}

Answer:
"""
        
        # Generate answer
        response = self.generate(prompt, **kwargs)
        
        # Extract the answer part
        answer_start = response.find("Answer:")
        if answer_start != -1:
            answer = response[answer_start + 7:].strip()
        else:
            answer = response.replace(prompt, "").strip()
            
        return answer 