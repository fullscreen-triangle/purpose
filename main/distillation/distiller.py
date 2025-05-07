import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import random
import sys

from main.base_models.llama_model import LlamaModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model modules

# For API-based models
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class KnowledgeDistiller:
    """
    Class for knowledge distillation from teacher models (OpenAI, Claude) to a student model (LLaMA).
    """
    
    def __init__(
        self,
        output_dir: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        teacher_models: Optional[List[str]] = None,
        local_model_path: Optional[str] = None,
    ):
        """
        Initialize the KnowledgeDistiller with API keys and output directory.
        
        Args:
            output_dir (str): Directory to save generated datasets and models
            openai_api_key (Optional[str]): OpenAI API key for using OpenAI models
            anthropic_api_key (Optional[str]): Anthropic API key for using Claude models
            teacher_models (Optional[List[str]]): List of teacher model names, defaults to ["gpt-4", "claude-3-opus-20240229"]
            local_model_path (Optional[str]): Path to local LLaMA model for inference
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set API keys 
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Set default teacher models if not provided
        if teacher_models is None:
            self.teacher_models = ["gpt-4", "claude-3-opus-20240229"]
        else:
            self.teacher_models = teacher_models
            
        # Initialize local LLaMA model if path is provided
        self.local_model = None
        self.local_model_path = local_model_path
        if local_model_path:
            logger.info(f"Local model path provided: {local_model_path}")
    
    def _get_client_for_model(self, model_name: str) -> Any:
        """
        Get the appropriate client for the specified model.
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            Any: The client object for the specified model
            
        Raises:
            ValueError: If the model name is not supported or API key is missing
        """
        if model_name.startswith("gpt"):
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            return openai.OpenAI(api_key=self.openai_api_key)
        elif model_name.startswith("claude"):
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key is required for Claude models")
            return Anthropic(api_key=self.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _load_local_model(self) -> LlamaModel:
        """
        Load the local LLaMA model if not already loaded.
        
        Returns:
            LlamaModel: The loaded local model
            
        Raises:
            ValueError: If no local model path was provided during initialization
        """
        if not self.local_model_path:
            raise ValueError("No local model path provided")
            
        if not self.local_model:
            logger.info(f"Loading local LLaMA model from {self.local_model_path}")
            self.local_model = LlamaModel(
                model_path=self.local_model_path,
                load_in_8bit=True,  # Use 8-bit quantization for efficiency
                temperature=0.7,
                max_new_tokens=512
            )
            
        return self.local_model
        
    def generate_training_example(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate a training example using a specified model.
        
        Args:
            prompt (str): The input prompt
            model (str): Model to use for generation (e.g., "gpt-4", "claude-3-opus-20240229")
            system_prompt (Optional[str]): System instructions for the model
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            Dict[str, Any]: A dictionary containing the prompt, completion, model and metadata
        """
        start_time = time.time()
        
        try:
            if model.startswith("gpt"):
                client = self._get_client_for_model(model)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                completion = response.choices[0].message.content
                
            elif model.startswith("claude"):
                client = self._get_client_for_model(model)
                response = client.messages.create(
                    model=model,
                    system=system_prompt or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                completion = response.content[0].text
                
            else:
                raise ValueError(f"Unsupported model: {model}")
                
            elapsed_time = time.time() - start_time
            logger.info(f"Generated training example with {model} in {elapsed_time:.2f}s")
            
            return {
                "prompt": prompt,
                "completion": completion,
                "model": model,
                "metadata": {
                    "timestamp": time.time(),
                    "generation_time": elapsed_time,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating training example with {model}: {e}")
            raise
    
    def generate_dataset(
        self,
        prompts: List[str],
        output_file: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        alternate_teachers: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset from a list of prompts.
        
        Args:
            prompts (List[str]): List of prompts to generate completions for
            output_file (str): Path to save the generated dataset
            system_prompt (Optional[str]): System instructions for the models
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate per completion
            alternate_teachers (bool): Whether to alternate between teacher models
            
        Returns:
            List[Dict[str, Any]]: The generated dataset
        """
        dataset = []
        output_path = os.path.join(self.output_dir, output_file)
        
        logger.info(f"Generating dataset with {len(prompts)} prompts")
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating example {i+1}/{len(prompts)}")
            
            # Select model - alternate if specified, otherwise random
            if alternate_teachers:
                model = self.teacher_models[i % len(self.teacher_models)]
            else:
                model = random.choice(self.teacher_models)
                
            try:
                example = self.generate_training_example(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                dataset.append(example)
                
                # Save progress incrementally
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error generating example for prompt {i+1}: {e}")
                logger.error(f"Skipping prompt: {prompt[:100]}...")
                continue
                
        logger.info(f"Dataset generation complete. Saved to {output_path}")
        return dataset
    
    def format_for_llama_finetuning(
        self,
        dataset: List[Dict[str, Any]],
        format_type: str = "alpaca",
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format the dataset for LLaMA fine-tuning.
        
        Args:
            dataset (List[Dict[str, Any]]): The dataset to format
            format_type (str): The format type to use (alpaca, llama, chatml)
            output_file (Optional[str]): Path to save the formatted dataset
            
        Returns:
            List[Dict[str, Any]]: The formatted dataset
        """
        formatted_dataset = []
        
        for example in dataset:
            prompt = example["prompt"]
            completion = example["completion"]
            
            if format_type == "alpaca":
                formatted_example = {
                    "instruction": prompt,
                    "input": "",
                    "output": completion
                }
            elif format_type == "llama":
                formatted_example = {
                    "text": f"<s>[INST] {prompt} [/INST] {completion}</s>"
                }
            elif format_type == "chatml":
                formatted_example = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                }
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
            formatted_dataset.append(formatted_example)
            
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w') as f:
                json.dump(formatted_dataset, f, indent=2)
            logger.info(f"Formatted dataset saved to {output_path}")
            
        return formatted_dataset
    
    def run_local_inference(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Run inference using the local LLaMA model.
        
        Args:
            prompt (str): The input prompt
            system_prompt (Optional[str]): System instructions
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: The generated text
            
        Raises:
            ValueError: If no local model is available
        """
        if not self.local_model_path:
            raise ValueError("No local model available for inference")
            
        model = self._load_local_model()
        
        logger.info(f"Running local inference with prompt length: {len(prompt)}")
        start_time = time.time()
        
        try:
            response = model.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Local inference completed in {elapsed_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error during local inference: {e}")
            raise
    
    def compare_models(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, str]:
        """
        Compare responses from teacher models and local model on the same prompt.
        
        Args:
            prompt (str): The input prompt to compare responses for
            system_prompt (Optional[str]): System instructions
            save_results (bool): Whether to save comparison results to a file
            
        Returns:
            Dict[str, str]: Dictionary mapping model names to their responses
        """
        results = {}
        
        # Get responses from teacher models
        for model in self.teacher_models:
            logger.info(f"Getting response from {model}")
            try:
                example = self.generate_training_example(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt
                )
                results[model] = example["completion"]
            except Exception as e:
                logger.error(f"Error getting response from {model}: {e}")
                results[model] = f"ERROR: {str(e)}"
        
        # Get response from local model if available
        if self.local_model_path:
            logger.info("Getting response from local LLaMA model")
            try:
                local_response = self.run_local_inference(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                results["local_llama"] = local_response
            except Exception as e:
                logger.error(f"Error getting response from local model: {e}")
                results["local_llama"] = f"ERROR: {str(e)}"
        
        # Save results if requested
        if save_results:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"comparison_{timestamp}.json")
            with open(output_path, 'w') as f:
                json.dump({
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "responses": results
                }, f, indent=2)
            logger.info(f"Comparison results saved to {output_path}")
        
        return results 