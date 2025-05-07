import logging
import os
import time
from typing import Optional, Union, Dict, List, Any, Generator
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

class LlamaModel:
    """
    Class for running inference with LLaMA model locally.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ):
        """
        Initialize the LLaMA model.
        
        Args:
            model_path (str): Path to the LLaMA model or model name from Hugging Face
            device (Optional[str]): Device to load the model on ('cpu', 'cuda', 'mps')
                If None, will use CUDA if available, else CPU
            load_in_8bit (bool): Whether to load the model in 8-bit precision
            load_in_4bit (bool): Whether to load the model in 4-bit precision
            max_new_tokens (int): Default maximum number of new tokens to generate
            temperature (float): Default temperature for generation
            top_p (float): Default top-p for nucleus sampling
            top_k (int): Default top-k for sampling
            repetition_penalty (float): Default repetition penalty
        """
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Quantization settings
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        if load_in_8bit and load_in_4bit:
            logger.warning("Both load_in_8bit and load_in_4bit are set to True. Using 4-bit precision.")
            self.load_in_8bit = False
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Model and tokenizer will be loaded on first use
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized LlamaModel with model path: {model_path}")
        logger.info(f"Device: {self.device}, 8-bit: {load_in_8bit}, 4-bit: {load_in_4bit}")
    
    def _load_model(self):
        """
        Load the model and tokenizer.
        """
        if self.model is not None and self.tokenizer is not None:
            return
        
        start_time = time.time()
        logger.info(f"Loading LLaMA model from {self.model_path}")
        
        # Configure quantization if needed
        quantization_config = None
        
        if self.load_in_4bit:
            logger.info("Loading model in 4-bit precision")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.load_in_8bit:
            logger.info("Loading model in 8-bit precision")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load tokenizer first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate configuration
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if self.device in ["cuda", "mps"] else "cpu",
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from the model.
        
        Args:
            prompt (str): The input prompt
            system_prompt (Optional[str]): System prompt for chat models
            max_new_tokens (Optional[int]): Maximum number of new tokens
            temperature (Optional[float]): Temperature for sampling
            top_p (Optional[float]): Top-p for nucleus sampling
            top_k (Optional[int]): Top-k for sampling
            repetition_penalty (Optional[float]): Repetition penalty
            stream (bool): Whether to stream the output
            
        Returns:
            Union[str, Generator[str, None, None]]: Generated text or streaming generator
        """
        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self._load_model()
        
        # Use default parameters if not specified
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        repetition_penalty = repetition_penalty or self.repetition_penalty
        
        # Prepare the input
        full_prompt = prompt
        if system_prompt:
            # Format with system prompt for chat-oriented models
            if "llama" in self.model_path.lower():
                # LLaMA 2 chat format
                full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                # Generic format
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        logger.info(f"Generating with prompt length: {len(full_prompt)}")
        logger.debug(f"Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        # Use streaming if requested
        if stream:
            return self._generate_stream(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
        
        # Generate text
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
        
        # Decode and strip the prompt
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the original prompt to get only the generated text
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
        
        return generated_text
    
    def _generate_stream(
        self,
        input_ids,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming output.
        
        Args:
            input_ids: Tokenized input
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            top_p (float): Top-p for nucleus sampling
            top_k (int): Top-k for sampling
            repetition_penalty (float): Repetition penalty
            
        Yields:
            str: Chunks of generated text
        """
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Run generation in a separate thread
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield from streamer
        for text in streamer:
            yield text
    
    def unload(self):
        """
        Unload the model and tokenizer to free up memory.
        """
        if self.model is not None:
            try:
                # Move model to CPU first if it's on GPU
                if self.device != "cpu":
                    self.model = self.model.to("cpu")
                
                # Delete model and clear CUDA cache
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = None
                logger.info("Model unloaded from memory")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            logger.info("Tokenizer unloaded from memory") 