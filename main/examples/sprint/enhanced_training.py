#!/usr/bin/env python3
"""
Enhanced Model Training Pipeline

This script demonstrates how to combine the knowledge distillation and 
model optimization approaches to create a high-quality domain-specific model.

The pipeline consists of three main phases:
1. Model Optimization: Generate formal mathematical models of sprint phenomena
2. Knowledge Distillation: Generate Q&A pairs using large LLMs
3. Combined Training: Train on both datasets

Usage:
    python enhanced_training.py --api-keys-from-env
"""

import os
import argparse
import logging
from pathlib import Path

from main.examples.sprint.knowledge_distill import KnowledgeDistiller
from main.examples.sprint.model_optimization import ModelOptimizer
from main.cli.runner import run_train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced-training")

# Constants
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DISTILL_CORPUS_DIR = DATA_DIR / "distill_corpus"
MODEL_CORPUS_DIR = DATA_DIR / "model_corpus"
COMBINED_CORPUS_DIR = DATA_DIR / "combined_corpus"


def create_combined_corpus():
    """
    Combine the knowledge distillation corpus and model optimization corpus
    into a single training corpus.
    """
    logger.info("Creating combined training corpus")
    
    # Ensure output directory exists
    os.makedirs(COMBINED_CORPUS_DIR, exist_ok=True)
    
    # Paths to source corpora
    distill_corpus = DISTILL_CORPUS_DIR / "qa_corpus.txt"
    model_corpus = MODEL_CORPUS_DIR / "model_corpus.txt"
    
    # Path to combined corpus
    combined_corpus = COMBINED_CORPUS_DIR / "combined_corpus.txt"
    
    # Check if source corpora exist
    if not distill_corpus.exists() or not model_corpus.exists():
        logger.error("Source corpora not found. Run knowledge distillation and model optimization first.")
        return None
    
    # Combine the corpora
    with open(combined_corpus, "w") as outfile:
        # First add model corpus (formal models)
        with open(model_corpus, "r") as infile:
            outfile.write(infile.read())
            outfile.write("\n\n" + "=" * 80 + "\n\n")
        
        # Then add QA corpus
        with open(distill_corpus, "r") as infile:
            outfile.write(infile.read())
    
    logger.info(f"Combined corpus created at {combined_corpus}")
    return str(combined_corpus)


def run_enhanced_pipeline(args):
    """Run the complete enhanced training pipeline."""
    # Step 1: Run model optimization if requested
    if args.run_model_optimization:
        logger.info("Starting model optimization phase")
        optimizer = ModelOptimizer(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            target_model=args.target_model,
            domain="sprint",
            model_type=args.model_type,
            output_dir=str(MODEL_CORPUS_DIR),
            num_samples=args.num_samples
        )
        optimizer.run_pipeline()
    
    # Step 2: Run knowledge distillation if requested
    if args.run_knowledge_distill:
        logger.info("Starting knowledge distillation phase")
        distiller = KnowledgeDistiller(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            target_model=args.target_model,
            domain="sprint",
            output_dir=str(DISTILL_CORPUS_DIR),
            num_samples=args.num_samples
        )
        distiller.run_pipeline()
    
    # Step 3: Create combined corpus
    if args.create_combined:
        combined_corpus_path = create_combined_corpus()
        if not combined_corpus_path:
            logger.error("Failed to create combined corpus")
            return
    
    # Step 4: Train on combined corpus
    if args.train_model:
        logger.info("Training model on combined corpus")
        run_train(
            data_dir=str(COMBINED_CORPUS_DIR),
            model_dir=str(MODEL_DIR),
            model_name=args.base_model,
            output_name=args.output_model_name,
            epochs=args.training_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_lora=True,
            memory_fraction=0.8
        )
    
    logger.info("Enhanced training pipeline complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Training Pipeline")
    
    # Pipeline control arguments
    parser.add_argument("--run-model-optimization", action="store_true",
                        help="Run the model optimization phase")
    parser.add_argument("--run-knowledge-distill", action="store_true",
                        help="Run the knowledge distillation phase") 
    parser.add_argument("--create-combined", action="store_true",
                        help="Create combined corpus from both approaches")
    parser.add_argument("--train-model", action="store_true",
                        help="Train model on the combined corpus")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all phases of the pipeline")
    
    # Model generation arguments
    parser.add_argument("--target-model", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"],
                        help="Target model to obtain knowledge from")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to generate in each phase")
    parser.add_argument("--model-type", type=str, default="mathematical",
                        choices=["mathematical", "statistical", "biomechanical"],
                        help="Type of models to generate in optimization phase")
    parser.add_argument("--api-keys-from-env", action="store_true",
                        help="Use API keys from environment variables")
    
    # Training arguments
    parser.add_argument("--base-model", type=str, default="gpt2",
                        help="Base model to fine-tune")
    parser.add_argument("--output-model-name", type=str, default="enhanced_sprint_model",
                        help="Name of the output model")
    parser.add_argument("--training-steps", type=int, default=3000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    args = parser.parse_args()
    
    # If run_all is specified, enable all phases
    if args.run_all:
        args.run_model_optimization = True
        args.run_knowledge_distill = True
        args.create_combined = True
        args.train_model = True
    
    return args


if __name__ == "__main__":
    args = parse_args()
    run_enhanced_pipeline(args) 