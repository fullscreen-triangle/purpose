#!/usr/bin/env python3
"""
D3 Visualization Model Generator

This script implements a specialized model for generating D3.js visualizations
in React for sprint performance models and their results.

The key idea is to train a model that can translate mathematical sprint models
and their outputs into interactive visualizations that can be integrated directly
into a React-based RAG system frontend.

Usage:
    python -m purpose.examples.visualization.d3_model_generator --num-samples 100
"""

import os
import json
import random
import argparse
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

# Import for model APIs
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("d3_visualization_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("d3-visualization")

# Constants
D3_EXAMPLES_DIR = "visualization_data"
D3_CORPUS_DIR = "data/d3_corpus"
OUTPUT_VIZ_MODEL_DIR = "models/visualization_model"


class D3VisualizationGenerator:
    """
    Generates training data for a D3 visualization model that can translate
    sprint performance models and results into interactive D3 visualizations in React.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        target_model: str = "gpt-4", 
        output_dir: str = D3_CORPUS_DIR,
        num_samples: int = 50,
        model_data_dir: str = "model_data"  # Directory containing the mathematical models
    ):
        """
        Initialize the D3 visualization generator.
        
        Args:
            openai_api_key: OpenAI API key (from env if not provided)
            anthropic_api_key: Anthropic API key (from env if not provided)
            target_model: Model to use for generating visualization code
            output_dir: Directory to save generated D3 examples
            num_samples: Number of visualization examples to generate
            model_data_dir: Directory containing mathematical sprint models
        """
        self.target_model = target_model
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.model_data_dir = model_data_dir
        
        # Setup API clients
        self.openai_client = None
        self.anthropic_client = None
        
        if openai_api_key or os.environ.get("OPENAI_API_KEY"):
            openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI(api_key=openai.api_key)
            logger.info(f"OpenAI client initialized")
        
        if anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            logger.info(f"Anthropic client initialized")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(D3_EXAMPLES_DIR, exist_ok=True)
        
        # Load visualization types and templates
        self._load_visualization_templates()
        
        # Load any existing sprint models if available
        self.sprint_models = self._load_sprint_models()
    
    def _load_sprint_models(self) -> List[Dict[str, Any]]:
        """Load existing sprint models from the model_data directory."""
        models = []
        
        try:
            # Try to load the complete models file if it exists
            complete_models_path = os.path.join(self.model_data_dir, "models_complete.json")
            if os.path.exists(complete_models_path):
                with open(complete_models_path, "r") as f:
                    models = json.load(f)
                    logger.info(f"Loaded {len(models)} sprint models from {complete_models_path}")
                return models
            
            # Otherwise, load individual model files
            model_files = [f for f in os.listdir(self.model_data_dir) if f.endswith(".json") and not f.startswith("models_")]
            for file in model_files:
                with open(os.path.join(self.model_data_dir, file), "r") as f:
                    model_data = json.load(f)
                    models.append(model_data)
            
            logger.info(f"Loaded {len(models)} sprint models from individual files")
            
        except Exception as e:
            logger.warning(f"Error loading sprint models: {str(e)}")
            logger.info("Will use synthetic model data instead")
            
        return models
    
    def _load_visualization_templates(self):
        """Load D3 visualization templates and types."""
        self.visualization_templates = [
            {
                "name": "time_velocity_curve",
                "description": "Create a D3.js visualization in React for a sprint velocity-time curve, showing acceleration, maximum velocity, and deceleration phases.",
                "model_type": "sprint_velocity_model",
                "chart_type": "line_chart",
                "axes": ["time (s)", "velocity (m/s)"],
                "features": ["zoom", "tooltip", "phase_markers"]
            },
            {
                "name": "stride_frequency_length",
                "description": "Create a D3.js visualization in React showing the relationship between stride frequency and stride length throughout a sprint race.",
                "model_type": "stride_parameters_model",
                "chart_type": "scatter_plot",
                "axes": ["stride frequency (Hz)", "stride length (m)"],
                "features": ["color_by_phase", "regression_line", "tooltip"]
            },
            {
                "name": "ground_reaction_force",
                "description": "Create a D3.js visualization in React for ground reaction forces during different phases of a sprint.",
                "model_type": "ground_reaction_force_model",
                "chart_type": "area_chart",
                "axes": ["time (s)", "force (N)"],
                "features": ["stacked_forces", "annotations", "zoom"]
            },
            {
                "name": "energy_expenditure_breakdown",
                "description": "Create a D3.js visualization in React showing the breakdown of energy systems during a sprint race.",
                "model_type": "energy_expenditure_model",
                "chart_type": "stacked_area_chart",
                "axes": ["race_progress (%)", "energy_contribution (%)"],
                "features": ["tooltip", "legend", "annotations"]
            },
            {
                "name": "sprint_start_optimization",
                "description": "Create a D3.js visualization in React for optimizing sprint start parameters including block angles and force application.",
                "model_type": "sprint_start_model",
                "chart_type": "heat_map",
                "axes": ["block_angle (degrees)", "force_application (N)"],
                "features": ["interactive_slider", "optimization_marker", "tooltip"]
            },
            {
                "name": "performance_prediction",
                "description": "Create a D3.js visualization in React for sprint performance prediction based on various athlete parameters.",
                "model_type": "performance_prediction_model",
                "chart_type": "radar_chart",
                "axes": ["training_volume", "muscle_fiber_composition", "strength_metrics", "previous_performance"],
                "features": ["comparison_profiles", "threshold_markers", "interactive_inputs"]
            },
            {
                "name": "race_strategy_optimizer",
                "description": "Create a D3.js visualization in React for optimizing race strategy across different segments of a sprint race.",
                "model_type": "race_strategy_optimization_model",
                "chart_type": "multi_line_chart",
                "axes": ["race_distance (m)", "velocity (m/s)"],
                "features": ["strategy_comparison", "optimal_path", "interactive_parameters"]
            },
            {
                "name": "biomechanical_efficiency",
                "description": "Create a D3.js visualization in React showing biomechanical efficiency metrics throughout a sprint.",
                "model_type": "biomechanical_efficiency_model",
                "chart_type": "parallel_coordinates",
                "axes": ["joint_angles", "ground_contact_time", "vertical_oscillation", "arm_movement"],
                "features": ["efficiency_score", "comparison_to_ideal", "filtering"]
            },
            {
                "name": "training_adaptation_curve",
                "description": "Create a D3.js visualization in React showing training adaptations over time for a sprint athlete.",
                "model_type": "sprint_training_adaptation_model",
                "chart_type": "area_chart",
                "axes": ["time (weeks)", "performance_gain (%)"],
                "features": ["training_markers", "plateau_detection", "supercompensation_periods"]
            },
            {
                "name": "wind_effect_analysis",
                "description": "Create a D3.js visualization in React showing the effects of wind on sprint performance at different speeds and altitudes.",
                "model_type": "wind_effect_model",
                "chart_type": "3d_surface",
                "axes": ["wind_speed (m/s)", "altitude (m)", "time_effect (s)"],
                "features": ["interactive_rotation", "cross_section_slider", "color_gradient"]
            }
        ]
        
        self.chart_types = {
            "line_chart": "A line chart showing the relationship between x and y variables over a continuous domain.",
            "scatter_plot": "A scatter plot showing individual data points to reveal patterns and correlations.",
            "area_chart": "An area chart showing quantitative data graphically, with the area between axis and line filled with color.",
            "stacked_area_chart": "A stacked area chart showing multiple series stacked on top of each other.",
            "heat_map": "A heat map using color to represent data values in a two-dimensional grid.",
            "radar_chart": "A radar chart displaying multivariate data as a two-dimensional chart with three or more variables on axes from the center.",
            "multi_line_chart": "A multi-line chart showing multiple series as separate lines for comparison.",
            "parallel_coordinates": "A parallel coordinates plot for visualizing high-dimensional multivariate data.",
            "3d_surface": "A 3D surface plot showing a functional relationship between two independent variables and one dependent variable."
        }
        
        self.system_prompt = """You are an expert D3.js and React developer specializing in creating interactive visualizations for sports science and sprint performance data.

Your task is to create complete, functional React components that implement D3.js visualizations for sprint performance models and their results.

For each visualization you create, you MUST:

1. Provide a COMPLETE React component, including all imports, component setup, D3 implementation, and exports.
2. Ensure your code follows modern React practices (functional components, hooks for D3 integration).
3. Implement proper axis labels, titles, and legends with appropriate units.
4. Add interactive features like tooltips, zooming, and parameter controls.
5. Include sample data that demonstrates the visualization, based on the mathematical model.
6. Add comments explaining key aspects of the visualization implementation.
7. Ensure the visualization is responsive and works well in different screen sizes.
8. Use a consistent, readable style with intuitive color schemes.

The component should be ready to integrate into a React application with minimal modifications.

Assume that the component will receive model results as props and should visualize those results appropriately."""
    
    def _generate_synthetic_model_data(self, model_type: str) -> Dict[str, Any]:
        """Generate synthetic data for a model if real models aren't available."""
        if model_type == "sprint_velocity_model":
            return {
                "model_name": "sprint_velocity_model",
                "formulation": """
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
            }
        elif model_type == "stride_parameters_model":
            return {
                "model_name": "stride_parameters_model",
                "formulation": """
                # Stride Parameters Model
                
                The relationship between stride length (SL), stride frequency (SF), and velocity (v) is:
                
                v = SL * SF
                
                During acceleration phase (0-30m):
                SF increases rapidly from 3.5 to 4.5 Hz
                SL increases gradually from 1.2 to 2.0m
                
                During maximum velocity phase (30-60m):
                SF reaches maximum of 4.5-5.0 Hz
                SL reaches maximum of 2.2-2.5m
                
                During deceleration phase (80-100m):
                SF decreases slightly
                SL decreases slightly
                """
            }
        else:
            # Generic data for other model types
            return {
                "model_name": model_type,
                "formulation": f"Mathematical formulation for {model_type} with sample parameters and equations."
            }
    
    def _generate_visualization_prompt(self) -> str:
        """Generate a prompt to create a D3.js visualization in React."""
        template = random.choice(self.visualization_templates)
        
        # Find a matching sprint model if available
        matching_model = None
        if self.sprint_models:
            matching_models = [m for m in self.sprint_models if m.get("model_name") == template["model_type"]]
            if matching_models:
                matching_model = random.choice(matching_models)
        
        # If no matching model found, generate synthetic model data
        if not matching_model:
            matching_model = self._generate_synthetic_model_data(template["model_type"])
        
        # Construct the prompt
        prompt = f"""
VISUALIZATION TASK: {template["description"]}

CHART TYPE: {template["chart_type"]} - {self.chart_types.get(template["chart_type"], "")}

AXES: 
- X-axis: {template["axes"][0]}
- Y-axis: {template["axes"][1]}
{f"- Z-axis: {template['axes'][2]}" if len(template["axes"]) > 2 else ""}

FEATURES TO IMPLEMENT:
{', '.join(template["features"])}

MODEL TO VISUALIZE:
{matching_model.get("formulation", f"Mathematical model for {template['model_type']}")}

Please create a complete React component that implements this D3.js visualization. The component should:
1. Be ready to integrate into a React application
2. Include sample data based on the model above
3. Implement all the requested features
4. Include appropriate axis labels, legends, and titles
5. Follow best practices for D3 integration with React

Return the full, functional React component code.
        """
        
        return prompt, template["name"]
    
    def _query_large_model(self, prompt: str) -> str:
        """Get a D3 visualization implementation from the target large language model."""
        try:
            if self.target_model.startswith("gpt") and self.openai_client:
                # OpenAI models
                response = self.openai_client.chat.completions.create(
                    model=self.target_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Lower temperature for more consistent code
                    max_tokens=4096   # Visualizations can be complex, need more tokens
                )
                return response.choices[0].message.content
                
            elif self.target_model.startswith("claude") and self.anthropic_client:
                # Anthropic models
                response = self.anthropic_client.messages.create(
                    model=self.target_model,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                return response.content[0].text
                
            else:
                logger.error(f"Unsupported model: {self.target_model}")
                return ""
                
        except Exception as e:
            logger.error(f"Error querying model: {str(e)}")
            return ""
    
    def generate_d3_dataset(self) -> List[Dict[str, Any]]:
        """Generate a dataset of D3 visualization implementations for sprint models."""
        dataset = []
        
        logger.info(f"Generating {self.num_samples} D3 visualization examples using {self.target_model}")
        
        for i in range(self.num_samples):
            viz_prompt, viz_name = self._generate_visualization_prompt()
            logger.info(f"Example {i+1}/{self.num_samples}: {viz_name}")
            
            d3_implementation = self._query_large_model(viz_prompt)
            
            if d3_implementation:
                example = {
                    "visualization_name": viz_name,
                    "prompt": viz_prompt,
                    "implementation": d3_implementation,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                dataset.append(example)
                
                # Save individual example file
                example_filename = f"{viz_name}_{i+1}.json"
                with open(os.path.join(D3_EXAMPLES_DIR, example_filename), "w") as f:
                    json.dump(example, f, indent=2)
                
                # Save intermediate results
                if (i + 1) % 5 == 0 or i == 0:
                    with open(f"{D3_EXAMPLES_DIR}/d3_examples_partial_{i+1}.json", "w") as f:
                        json.dump(dataset, f, indent=2)
                
                # Avoid rate limits
                time.sleep(2)
            else:
                logger.warning(f"Failed to get D3 implementation for: {viz_name}")
        
        # Save complete dataset
        with open(f"{D3_EXAMPLES_DIR}/d3_examples_complete.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Generated {len(dataset)} D3 visualization examples")
        return dataset
    
    def create_d3_corpus(self, dataset: List[Dict[str, Any]]) -> str:
        """Convert D3 visualization dataset to a training corpus format."""
        corpus_path = os.path.join(self.output_dir, "d3_visualization_corpus.txt")
        jsonl_path = os.path.join(self.output_dir, "d3_visualization_data.jsonl")
        
        # Create training corpus text file
        with open(corpus_path, "w") as f:
            for item in dataset:
                f.write(f"VISUALIZATION TASK: {item['visualization_name']}\n\n")
                f.write(f"PROMPT: {item['prompt']}\n\n")
                f.write(f"IMPLEMENTATION:\n{item['implementation']}\n\n")
                f.write("=" * 80 + "\n\n")
        
        # Create JSONL file for structured access
        with open(jsonl_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Created D3 visualization corpus at {corpus_path}")
        logger.info(f"Created JSONL dataset at {jsonl_path}")
        
        return corpus_path
    
    def train_visualization_model(self, corpus_path: str, output_dir: str = OUTPUT_VIZ_MODEL_DIR):
        """Train a model on the D3 visualization corpus."""
        logger.info(f"Training model on D3 visualization corpus")
        
        # Initialize training process (in a real implementation, this would use your existing training code)
        logger.info(f"Would train model using corpus at {corpus_path} and save to {output_dir}")
        
        # In a real implementation, you would call your existing training code here
        
        logger.info(f"Visualization model training complete")
    
    def run_pipeline(self):
        """Run the complete D3 visualization generation pipeline."""
        # Step 1: Generate D3 visualization dataset
        dataset = self.generate_d3_dataset()
        
        # Step 2: Create visualization corpus
        corpus_path = self.create_d3_corpus(dataset)
        
        # Step 3: Train model on D3 visualization corpus
        self.train_visualization_model(corpus_path)
        
        logger.info("D3 visualization generation pipeline complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="D3 Visualization Generator Pipeline")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of D3 visualization examples to generate")
    parser.add_argument("--target-model", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"],
                        help="Target model to generate visualizations from")
    parser.add_argument("--model-data-dir", type=str, default="model_data",
                        help="Directory containing mathematical sprint models")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key (will use env var if not provided)")
    parser.add_argument("--anthropic-key", type=str, default=None,
                        help="Anthropic API key (will use env var if not provided)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generator = D3VisualizationGenerator(
        openai_api_key=args.openai_key,
        anthropic_api_key=args.anthropic_key,
        target_model=args.target_model,
        model_data_dir=args.model_data_dir,
        num_samples=args.num_samples
    )
    
    generator.run_pipeline() 