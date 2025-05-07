import logging
import os
from typing import Dict, Any, List, Optional, Union

from ..inference.llama_inference import LlamaInference
# I asked you to use specialised models for all the processes.

from ..utils.file_utils import save_json, load_json, ensure_directory_exists

logger = logging.getLogger(__name__)

class LlamaDistillationPipeline:
    """
    Pipeline for using LLaMA as the base model, but OpenAI and Claude for knowledge distillation.
    This allows for local inference while leveraging more powerful models for training data generation.
    """
    
    def __init__(
        self, 
        llama_model_path: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        llama_bit_precision: int = 4,
        device_map: str = "auto",
        output_dir: str = "output/llama_distillation",
        openai_model: str = "gpt-4o",
        anthropic_model: str = "claude-3-opus-20240229",
        **model_kwargs
    ):
        """
        Initialize the LLaMA distillation pipeline.
        
        Args:
            llama_model_path: Path to the LLaMA model for inference
            openai_api_key: OpenAI API key (for knowledge distillation)
            anthropic_api_key: Anthropic API key (for knowledge distillation)
            llama_bit_precision: Bit precision for LLaMA model quantization
            device_map: Device mapping strategy for LLaMA
            output_dir: Directory to save outputs
            openai_model: OpenAI model to use for knowledge distillation
            anthropic_model: Anthropic model to use for knowledge distillation
            **model_kwargs: Additional model parameters
        """
        self.output_dir = output_dir
        ensure_directory_exists(output_dir)
        
        # Initialize LLaMA model for inference
        logger.info(f"Initializing LLaMA model from {llama_model_path}")
        self.llama = LlamaInference(
            model_path=llama_model_path,
            bit_precision=llama_bit_precision,
            device_map=device_map,
            **model_kwargs
        )
        
        # Initialize knowledge distillation models if API keys are provided
        self.openai = None
        self.anthropic = None
        # the two inference engines below don't exist. I also asked you to use specialised models
        if openai_api_key:
            logger.info(f"Initializing OpenAI model {openai_model} for knowledge distillation")
            self.openai = OpenAIInference(
                api_key=openai_api_key,
                model=openai_model
            )
        
        if anthropic_api_key:
            logger.info(f"Initializing Anthropic model {anthropic_model} for knowledge distillation")
            self.anthropic = AnthropicInference(
                api_key=anthropic_api_key,
                model=anthropic_model
            )
        
        if not (self.openai or self.anthropic):
            logger.warning("No knowledge distillation models initialized. Provide at least one of OpenAI or Anthropic API keys.")
        
        # Log model information
        llama_info = self.llama.get_model_info()
        logger.info(f"LLaMA model loaded: {llama_info.get('model_type', 'unknown')} with {llama_info.get('num_parameters', 'unknown')} parameters")
    
    def generate_with_llama(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate using the local LLaMA model.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system instructions
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        return self.llama.generate(prompt, system_prompt, **kwargs)
    
    def generate_knowledge_distillation_data(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        use_openai: bool = True,
        use_anthropic: bool = True,
        save_results: bool = True,
        output_file: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate knowledge distillation data using OpenAI and/or Anthropic models.
        
        Args:
            prompts: List of input prompts to generate training data for
            system_prompt: Optional system instructions
            use_openai: Whether to use OpenAI for distillation
            use_anthropic: Whether to use Anthropic for distillation
            save_results: Whether to save the results to disk
            output_file: Path to save the results (relative to output_dir)
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries containing the generated distillation data
        """
        if not (use_openai or use_anthropic):
            raise ValueError("At least one of use_openai or use_anthropic must be True")
        
        if use_openai and not self.openai:
            raise ValueError("OpenAI API key not provided during initialization")
        
        if use_anthropic and not self.anthropic:
            raise ValueError("Anthropic API key not provided during initialization")
        
        # Initialize results container
        distillation_data = []
        
        # Generate with OpenAI
        if use_openai and self.openai:
            logger.info(f"Generating knowledge distillation data with OpenAI for {len(prompts)} prompts")
            for i, prompt in enumerate(prompts):
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                
                # Generate with OpenAI
                openai_result = self.openai.generate(prompt, system_prompt, **kwargs)
                
                # Store the result
                distillation_item = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "openai_completion": openai_result.get("generated_text", ""),
                    "model_info": {
                        "openai_model": self.openai.model
                    }
                }
                
                distillation_data.append(distillation_item)
        
        # Generate with Anthropic
        if use_anthropic and self.anthropic:
            logger.info(f"Generating knowledge distillation data with Anthropic for {len(prompts)} prompts")
            
            # If we already have data from OpenAI, just add Anthropic completions
            if distillation_data:
                for i, (item, prompt) in enumerate(zip(distillation_data, prompts)):
                    logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                    
                    # Generate with Anthropic
                    anthropic_result = self.anthropic.generate(prompt, system_prompt, **kwargs)
                    
                    # Add to existing data
                    item["anthropic_completion"] = anthropic_result.get("generated_text", "")
                    item["model_info"]["anthropic_model"] = self.anthropic.model
            else:
                # Generate new data with just Anthropic
                for i, prompt in enumerate(prompts):
                    logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                    
                    # Generate with Anthropic
                    anthropic_result = self.anthropic.generate(prompt, system_prompt, **kwargs)
                    
                    # Store the result
                    distillation_item = {
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "anthropic_completion": anthropic_result.get("generated_text", ""),
                        "model_info": {
                            "anthropic_model": self.anthropic.model
                        }
                    }
                    
                    distillation_data.append(distillation_item)
        
        # Save results if requested
        if save_results:
            output_path = output_file or f"distillation_data_{len(prompts)}_prompts.json"
            full_path = os.path.join(self.output_dir, output_path)
            save_json(distillation_data, full_path)
            logger.info(f"Knowledge distillation data saved to {full_path}")
        
        return distillation_data
    
    def evaluate_llama_vs_teachers(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        use_openai: bool = True,
        use_anthropic: bool = True,
        save_results: bool = True,
        output_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare LLaMA outputs against the teacher models (OpenAI/Anthropic).
        
        Args:
            prompts: List of prompts to compare on
            system_prompt: Optional system instructions
            use_openai: Whether to include OpenAI in comparison
            use_anthropic: Whether to include Anthropic in comparison
            save_results: Whether to save the comparison results
            output_file: Path to save the results (relative to output_dir)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with comparison results
        """
        if not (use_openai or use_anthropic):
            raise ValueError("At least one of use_openai or use_anthropic must be True")
        
        if use_openai and not self.openai:
            raise ValueError("OpenAI API key not provided during initialization")
        
        if use_anthropic and not self.anthropic:
            raise ValueError("Anthropic API key not provided during initialization")
        
        # Initialize results container
        comparison_data = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Comparing models on prompt {i+1}/{len(prompts)}")
            
            # Generate with LLaMA
            llama_result = self.llama.generate(prompt, system_prompt, **kwargs)
            
            # Initialize comparison item
            comparison_item = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "llama_completion": llama_result.get("generated_text", ""),
                "model_info": {
                    "llama_model": self.llama.model_path
                }
            }
            
            # Generate with OpenAI if requested
            if use_openai and self.openai:
                openai_result = self.openai.generate(prompt, system_prompt, **kwargs)
                comparison_item["openai_completion"] = openai_result.get("generated_text", "")
                comparison_item["model_info"]["openai_model"] = self.openai.model
            
            # Generate with Anthropic if requested
            if use_anthropic and self.anthropic:
                anthropic_result = self.anthropic.generate(prompt, system_prompt, **kwargs)
                comparison_item["anthropic_completion"] = anthropic_result.get("generated_text", "")
                comparison_item["model_info"]["anthropic_model"] = self.anthropic.model
            
            comparison_data.append(comparison_item)
        
        # Save results if requested
        if save_results:
            output_path = output_file or f"model_comparison_{len(prompts)}_prompts.json"
            full_path = os.path.join(self.output_dir, output_path)
            save_json(comparison_data, full_path)
            logger.info(f"Model comparison data saved to {full_path}")
        
        return {"comparisons": comparison_data, "num_prompts": len(prompts)} 