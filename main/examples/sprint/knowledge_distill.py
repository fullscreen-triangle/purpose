#!/usr/bin/env python3
"""
Knowledge Distillation Pipeline for Sprint Domain

This script implements knowledge distillation from larger models (GPT-4/Claude)
to create high-quality training data for domain-specific sprint models.

The process:
1. Generate synthetic data
2. Obtain expert responses from large models
3. Structure data for domain-specific training
4. Train the domain model on this curated data

Usage:
    python knowledge_distill.py --num-samples 100 --target-model gpt-4 --domain sprint
"""

import os
import json
import random
import argparse
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

# Import for model APIs - install these with pip
import openai
from anthropic import Anthropic

# Local imports
from purpose.trainer import TransformersTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_distillation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("knowledge-distillation")

# Constants
SYNTHETIC_DATA_DIR = "synthetic_data"
DISTILLED_CORPUS_DIR = "data/distilled"
OUTPUT_MODEL_DIR = "models/distilled_sprint_model"

class KnowledgeDistiller:
    """
    Implements knowledge distillation from large language models to domain-specific models.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        target_model: str = "gpt-4", 
        domain: str = "sprint",
        output_dir: str = DISTILLED_CORPUS_DIR,
        num_samples: int = 50,
    ):
        """
        Initialize the knowledge distiller.
        
        Args:
            openai_api_key: OpenAI API key (from env if not provided)
            anthropic_api_key: Anthropic API key (from env if not provided)
            target_model: Model to use (gpt-4, gpt-3.5-turbo, claude-3-sonnet, claude-3-opus)
            domain: Domain for specialization
            output_dir: Directory to save distilled knowledge
            num_samples: Number of synthetic samples to generate
        """
        self.domain = domain
        self.target_model = target_model
        self.output_dir = output_dir
        self.num_samples = num_samples
        
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
        os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
        
        # Load domain-specific generators
        self._load_domain_generators()
    
    def _load_domain_generators(self):
        """Load domain-specific data generators."""
        if self.domain == "sprint":
            # Sprint-specific generators
            self.question_templates = [
                "What are the optimal biomechanical parameters for a {height}cm, {weight}kg sprinter to maximize acceleration in the first 30 meters?",
                "How does stride frequency change throughout a 100m sprint race for elite athletes?",
                "Compare the technical differences between {athlete1} and {athlete2} in their sprint mechanics.",
                "What is the impact of a tailwind of {wind_speed}m/s on 100m sprint performance at {altitude}m altitude?",
                "Analyze the impact of {surface_type} track surface on sprint times compared to modern synthetic tracks.",
                "What physiological adaptations occur in elite sprinters' muscle fiber composition after {years} years of training?",
                "How does block angle and positioning affect reaction time and the first 10m of a sprint?",
                "What are the key differences in sprint technique when comparing athletes from {country1} and {country2}?",
                "Analyze the mathematical model for estimating maximum velocity based on an athlete's height, weight, and muscle fiber composition.",
                "What is the relationship between ground contact time and stride length in the maximum velocity phase?",
                "How has sprint spike technology evolved from {year1} to {year2}, and what impact has this had on performance?",
                "What are the biomechanical differences between male and female elite sprinters in the {race_type}?",
                "Explain how the distribution of energy expenditure should be optimized across different segments of a {race_distance}m race.",
                "What are the implications of {body_type} body structure on sprint performance?",
                "Analyze the historical progression of world records in the {race_type} and identify key technological and training breakthroughs."
            ]
            
            self.parameter_ranges = {
                "height": (165, 200),
                "weight": (60, 95),
                "athlete1": ["Usain Bolt", "Noah Lyles", "Sha'Carri Richardson", "Shelly-Ann Fraser-Pryce", 
                            "Christian Coleman", "Trayvon Bromell", "Elaine Thompson-Herah", "Justin Gatlin",
                            "Fred Kerley", "Asafa Powell", "Yohan Blake", "Shericka Jackson"],
                "athlete2": ["Usain Bolt", "Noah Lyles", "Sha'Carri Richardson", "Shelly-Ann Fraser-Pryce", 
                            "Christian Coleman", "Trayvon Bromell", "Elaine Thompson-Herah", "Justin Gatlin",
                            "Fred Kerley", "Asafa Powell", "Yohan Blake", "Shericka Jackson"],
                "wind_speed": (0.5, 2.0),
                "altitude": [0, 500, 1000, 1500, 2000, 2500],
                "surface_type": ["cinder", "clay", "grass", "indoor wooden", "concrete", "Mondo", "Rekortan"],
                "years": (2, 20),
                "country1": ["Jamaica", "USA", "Great Britain", "Canada", "Japan", "China", "South Africa", "Italy", 
                           "France", "Germany", "Kenya", "Nigeria"],
                "country2": ["Jamaica", "USA", "Great Britain", "Canada", "Japan", "China", "South Africa", "Italy", 
                           "France", "Germany", "Kenya", "Nigeria"],
                "year1": range(1950, 2010, 10),
                "year2": range(2010, 2024),
                "race_type": ["100m", "200m", "400m", "4x100m relay", "60m indoor"],
                "race_distance": [60, 100, 200, 400],
                "body_type": ["mesomorphic", "ectomorphic", "endomorphic", "high center of gravity", "low center of gravity"]
            }
            
            self.system_prompt = f"""You are an expert in track and field sprint biomechanics, physiology, and performance analysis. 
            Your task is to provide detailed, scientifically accurate responses to questions about sprinting.
            
            Your responses should:
            1. Incorporate specific numerical values, equations, and formulas where relevant
            2. Reference scientific literature and research findings
            3. Include specific details about biomechanical parameters
            4. Structure information using proper technical terminology 
            5. Think of the response as training data for a model that will become an expert in this domain
            
            Important: Do not simplify or generalize your answers. Instead, provide detailed, specialized knowledge
            with the technical precision that would be expected in a scientific paper or textbook on sprint science.
            
            Whenever possible, include mathematical formulations, specific ranges of optimal values, and model-based 
            explanations rather than just descriptions."""
        else:
            # Default generators for other domains
            self.question_templates = [
                "Explain the fundamentals of {domain}",
                "What are the key components of {domain}?",
                "How has {domain} evolved over time?",
                "What are the current best practices in {domain}?"
            ]
            self.parameter_ranges = {
                "domain": [self.domain]
            }
            self.system_prompt = f"You are an expert in {self.domain}. Provide detailed, specialized responses."
    
    def _generate_synthetic_question(self) -> str:
        """Generate a synthetic question based on templates."""
        template = random.choice(self.question_templates)
        
        # Replace placeholders with random values
        for param, value_range in self.parameter_ranges.items():
            if param in template:
                if isinstance(value_range, tuple) and len(value_range) == 2:
                    # Numeric range
                    if isinstance(value_range[0], int):
                        value = random.randint(value_range[0], value_range[1])
                    else:
                        value = round(random.uniform(value_range[0], value_range[1]), 1)
                elif isinstance(value_range, list) or isinstance(value_range, range):
                    # List of options
                    value = random.choice(value_range)
                else:
                    value = value_range
                
                template = template.replace(f"{{{param}}}", str(value))
        
        return template
    
    def _query_large_model(self, question: str) -> str:
        """Get a response from the target large language model."""
        try:
            if self.target_model.startswith("gpt") and self.openai_client:
                # OpenAI models
                response = self.openai_client.chat.completions.create(
                    model=self.target_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.2,  # Lower temperature for more factual responses
                    max_tokens=1024
                )
                return response.choices[0].message.content
                
            elif self.target_model.startswith("claude") and self.anthropic_client:
                # Anthropic models
                response = self.anthropic_client.messages.create(
                    model=self.target_model,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                return response.content[0].text
                
            else:
                logger.error(f"Unsupported model: {self.target_model}")
                return ""
                
        except Exception as e:
            logger.error(f"Error querying model: {str(e)}")
            return ""
    
    def generate_synthetic_dataset(self) -> List[Dict[str, str]]:
        """Generate a synthetic dataset of questions and expert answers."""
        dataset = []
        
        logger.info(f"Generating {self.num_samples} synthetic samples using {self.target_model}")
        
        for i in range(self.num_samples):
            question = self._generate_synthetic_question()
            logger.info(f"Sample {i+1}/{self.num_samples}: {question}")
            
            answer = self._query_large_model(question)
            
            if answer:
                dataset.append({
                    "question": question,
                    "answer": answer
                })
                
                # Save intermediate results
                if (i + 1) % 10 == 0 or i == 0:
                    with open(f"{SYNTHETIC_DATA_DIR}/synthetic_data_partial_{i+1}.json", "w") as f:
                        json.dump(dataset, f, indent=2)
                
                # Avoid rate limits
                time.sleep(1)
            else:
                logger.warning(f"Failed to get answer for question: {question}")
        
        # Save complete dataset
        with open(f"{SYNTHETIC_DATA_DIR}/synthetic_data_complete.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Generated {len(dataset)} synthetic samples")
        return dataset
    
    def create_training_corpus(self, dataset: List[Dict[str, str]]) -> str:
        """Convert dataset to a training corpus format."""
        corpus_path = os.path.join(self.output_dir, "distilled_corpus.txt")
        jsonl_path = os.path.join(self.output_dir, "distilled_data.jsonl")
        
        # Create training corpus text file
        with open(corpus_path, "w") as f:
            for item in dataset:
                f.write(f"Question: {item['question']}\n\n")
                f.write(f"Answer: {item['answer']}\n\n")
                f.write("=" * 80 + "\n\n")
        
        # Create JSONL file for structured access
        with open(jsonl_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Created training corpus at {corpus_path}")
        logger.info(f"Created JSONL dataset at {jsonl_path}")
        
        return corpus_path
    
    def train_model(self, corpus_path: str, model_name: str = "gpt2", output_dir: str = OUTPUT_MODEL_DIR):
        """Train a model on the distilled corpus."""
        logger.info(f"Training model {model_name} on distilled corpus")
        
        # Create model directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize trainer 
        trainer = TransformersTrainer(
            model_name=model_name,
            output_dir=output_dir,
            corpus_path=corpus_path,
            batch_size=4,
            learning_rate=5e-5,
            num_epochs=3,
            use_lora=True
        )
        
        # Train the model
        trainer.train()
        
        logger.info(f"Model training complete. Model saved to {output_dir}")
    
    def run_pipeline(self):
        """Run the complete knowledge distillation pipeline."""
        # Step 1: Generate synthetic dataset
        dataset = self.generate_synthetic_dataset()
        
        # Step 2: Create training corpus
        corpus_path = self.create_training_corpus(dataset)
        
        # Step 3: Train model on distilled corpus
        self.train_model(corpus_path)
        
        logger.info("Knowledge distillation pipeline complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Pipeline")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--target-model", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"],
                        help="Target model to distill knowledge from")
    parser.add_argument("--domain", type=str, default="sprint",
                        help="Domain to specialize in")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key (will use env var if not provided)")
    parser.add_argument("--anthropic-key", type=str, default=None,
                        help="Anthropic API key (will use env var if not provided)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    distiller = KnowledgeDistiller(
        openai_api_key=args.openai_key,
        anthropic_api_key=args.anthropic_key,
        target_model=args.target_model,
        domain=args.domain,
        num_samples=args.num_samples
    )
    
    distiller.run_pipeline() 