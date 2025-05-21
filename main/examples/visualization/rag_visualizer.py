#!/usr/bin/env python3
"""
RAG Visualization Integration Module

This module connects the RAG system to the D3 visualization model,
allowing sprint model results to be automatically visualized in the frontend.

The system works as follows:
1. The RAG system retrieves and optimizes mathematical models for sprint phenomena
2. The D3 visualization model translates these models into React/D3 visualizations
3. The frontend displays both the model results and interactive visualizations

Usage:
    # Import and use in your RAG application
    from main.examples.visualization.rag_visualizer import RAGVisualizer
    
    visualizer = RAGVisualizer()
    visualization_code = visualizer.generate_visualization(model_result)
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
import re
from pathlib import Path

# Import transformers for model inference
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import for model APIs if using remote models instead of local
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag-visualizer")

# Constants
DEFAULT_MODEL_DIR = "models/visualization_model"
MODEL_PATTERN_REGEX = r"# Sprint (\w+) Model"  # Pattern to identify model types
SAMPLE_VISUALIZATION_DIR = "visualization_data"  # Fallback directory for sample visualizations


class RAGVisualizer:
    """
    Bridge between RAG system and D3 visualization model.
    This class takes sprint model results from the RAG system and
    generates corresponding D3.js visualizations for the frontend.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_DIR,
        use_local_model: bool = True,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        target_remote_model: str = "gpt-4",
        sample_visualizations_path: str = SAMPLE_VISUALIZATION_DIR
    ):
        """
        Initialize the RAG visualizer.
        
        Args:
            model_path: Path to the local visualization model, or name of remote model
            use_local_model: Whether to use a local model (True) or remote API (False)
            openai_api_key: OpenAI API key for remote inference (from env if not provided)
            anthropic_api_key: Anthropic API key for remote inference (from env if not provided)
            target_remote_model: Remote model to use if use_local_model is False
            sample_visualizations_path: Directory with sample visualizations for fallback
        """
        self.model_path = model_path
        self.use_local_model = use_local_model
        self.target_remote_model = target_remote_model
        self.sample_visualizations_path = sample_visualizations_path
        
        # Initialize model
        if use_local_model:
            try:
                logger.info(f"Loading local model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                logger.info("Local model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading local model: {str(e)}")
                logger.info("Falling back to remote model")
                self.use_local_model = False
        
        # Setup remote API clients if needed
        self.openai_client = None
        self.anthropic_client = None
        
        if not use_local_model:
            if openai_api_key or os.environ.get("OPENAI_API_KEY"):
                openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
                self.openai_client = openai.OpenAI(api_key=openai.api_key)
                logger.info(f"OpenAI client initialized")
            
            if anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
                self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                logger.info(f"Anthropic client initialized")
        
        # Load sample visualizations for fallback
        self.sample_visualizations = self._load_sample_visualizations()
    
    def _load_sample_visualizations(self) -> Dict[str, str]:
        """
        Load sample D3 visualizations to use as fallback options.
        Returns a dictionary mapping model types to visualization code.
        """
        visualizations = {}
        
        try:
            # Check for a complete examples file
            complete_file = os.path.join(self.sample_visualizations_path, "d3_examples_complete.json")
            if os.path.exists(complete_file):
                with open(complete_file, "r") as f:
                    examples = json.load(f)
                
                # Group by model type
                for example in examples:
                    viz_name = example.get("visualization_name", "")
                    if viz_name and "implementation" in example:
                        model_type = viz_name.split("_")[0]  # Extract base model type
                        visualizations[model_type] = example["implementation"]
                
                logger.info(f"Loaded {len(visualizations)} sample visualizations from complete file")
                return visualizations
            
            # Otherwise check for individual files
            viz_files = [f for f in os.listdir(self.sample_visualizations_path) 
                        if f.endswith(".json") and not f.startswith("d3_examples")]
            
            for file in viz_files:
                with open(os.path.join(self.sample_visualizations_path, file), "r") as f:
                    example = json.load(f)
                    if "visualization_name" in example and "implementation" in example:
                        model_type = example["visualization_name"].split("_")[0]
                        visualizations[model_type] = example["implementation"]
            
            logger.info(f"Loaded {len(visualizations)} sample visualizations from individual files")
            
        except Exception as e:
            logger.warning(f"Error loading sample visualizations: {str(e)}")
            
        return visualizations
    
    def _detect_model_type(self, model_result: str) -> str:
        """
        Detect the type of sprint model from the result text.
        
        Args:
            model_result: The sprint model result text
            
        Returns:
            The detected model type or "generic" if can't be determined
        """
        # Try to find model type using regex
        match = re.search(MODEL_PATTERN_REGEX, model_result)
        if match:
            return match.group(1).lower()
        
        # Check for key terms in the text
        model_indicators = {
            "velocity": "velocity",
            "stride": "stride",
            "ground reaction force": "ground_reaction",
            "energy expenditure": "energy",
            "sprint start": "start",
            "performance prediction": "performance",
            "race strategy": "strategy",
            "biomechanical efficiency": "biomechanical",
            "training adaptation": "training",
            "wind effect": "wind"
        }
        
        for term, model_type in model_indicators.items():
            if term.lower() in model_result.lower():
                return model_type
        
        return "generic"
    
    def _create_visualization_prompt(self, model_result: str, model_type: str) -> str:
        """
        Create a prompt to generate a D3 visualization for the model result.
        
        Args:
            model_result: The model result to visualize
            model_type: The detected model type
            
        Returns:
            A prompt for the visualization model
        """
        return f"""
VISUALIZATION REQUEST

MODEL RESULT:
{model_result}

MODEL TYPE: {model_type}

Please create a complete React component that implements a D3.js visualization for this model result. 
The component should:
1. Be ready to integrate into a React application
2. Include sample data based on the model result
3. Implement appropriate interactive features (tooltips, zoom, etc.)
4. Include proper axis labels, legends, and titles
5. Follow best practices for D3 integration with React

Return only the full, functional React component code.
        """
    
    def _get_local_model_visualization(self, prompt: str) -> str:
        """Generate visualization using local model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=4096,
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=1
            )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the component code from the result (might include some extra text)
            component_start = result.find("import React")
            if component_start == -1:
                component_start = result.find("import {")
            
            if component_start != -1:
                result = result[component_start:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating visualization with local model: {str(e)}")
            return ""
    
    def _get_remote_model_visualization(self, prompt: str) -> str:
        """Generate visualization using remote model API."""
        try:
            if self.target_remote_model.startswith("gpt") and self.openai_client:
                # OpenAI models
                system_prompt = """You are an expert D3.js and React developer. 
                Create complete, functional React components that visualize sprint model results.
                Your response should be ONLY the React component code, nothing else."""
                
                response = self.openai_client.chat.completions.create(
                    model=self.target_remote_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                return response.choices[0].message.content
                
            elif self.target_remote_model.startswith("claude") and self.anthropic_client:
                # Anthropic models
                system_prompt = """You are an expert D3.js and React developer. 
                Create complete, functional React components that visualize sprint model results.
                Your response should be ONLY the React component code, nothing else."""
                
                response = self.anthropic_client.messages.create(
                    model=self.target_remote_model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                return response.content[0].text
                
            else:
                logger.error(f"Unsupported remote model: {self.target_remote_model}")
                return ""
                
        except Exception as e:
            logger.error(f"Error querying remote model: {str(e)}")
            return ""
    
    def _get_fallback_visualization(self, model_type: str) -> str:
        """Get a fallback visualization based on model type."""
        # Look for an exact match
        if model_type in self.sample_visualizations:
            return self.sample_visualizations[model_type]
        
        # Look for a partial match
        for key, viz in self.sample_visualizations.items():
            if model_type in key or key in model_type:
                return viz
        
        # Return a generic visualization if available, otherwise empty string
        return self.sample_visualizations.get("generic", "")
    
    def generate_visualization(self, model_result: str) -> str:
        """
        Generate a D3.js visualization for the given model result.
        
        Args:
            model_result: The sprint model result to visualize
            
        Returns:
            React component code implementing the D3 visualization
        """
        logger.info("Generating visualization for model result")
        
        # Detect model type
        model_type = self._detect_model_type(model_result)
        logger.info(f"Detected model type: {model_type}")
        
        # Create prompt for the model
        prompt = self._create_visualization_prompt(model_result, model_type)
        
        # Generate visualization based on configuration
        if self.use_local_model:
            visualization = self._get_local_model_visualization(prompt)
        else:
            visualization = self._get_remote_model_visualization(prompt)
        
        # If generation failed, use fallback
        if not visualization or len(visualization) < 100:  # Arbitrary threshold to check for meaningful content
            logger.warning("Visualization generation failed, using fallback")
            visualization = self._get_fallback_visualization(model_type)
        
        return visualization
    
    def get_visualization_types(self) -> Dict[str, str]:
        """
        Get available visualization types.
        
        Returns:
            Dictionary mapping visualization types to descriptions
        """
        return {
            "velocity": "Sprint velocity-time curve",
            "stride": "Stride frequency-length relationship",
            "ground_reaction": "Ground reaction forces during sprint",
            "energy": "Energy expenditure breakdown",
            "start": "Sprint start optimization",
            "performance": "Performance prediction based on athlete parameters",
            "strategy": "Race strategy optimization",
            "biomechanical": "Biomechanical efficiency metrics",
            "training": "Training adaptation curve",
            "wind": "Wind effect analysis"
        }


if __name__ == "__main__":
    # Example usage
    visualizer = RAGVisualizer(use_local_model=False)  # Use remote model for this example
    
    # Example sprint model result
    example_result = """
    # Sprint Velocity Model
    
    The sprint velocity as a function of time is modeled as:
    
    v(t) = v_max * (1 - e^(-t/tau)) for t <= t_max
    v(t) = v_max * e^(-(t-t_max)/tau_d) for t > t_max
    
    where:
    - v_max is the maximum velocity (10-12 m/s for elite sprinters)
    - tau is the acceleration time constant (0.8-1.2s)
    - t_max is the time at which maximum velocity is reached (5-7s)
    - tau_d is the deceleration time constant (4-6s)
    
    Sample parameters:
    v_max = 11.2 m/s
    tau = 1.0s
    t_max = 6.0s
    tau_d = 5.0s
    """
    
    # Generate visualization
    visualization = visualizer.generate_visualization(example_result)
    
    # In a real application, you would return this to your frontend
    print(f"Generated D3 visualization with {len(visualization)} characters")
    print(visualization[:300] + "...")  # Print a preview 