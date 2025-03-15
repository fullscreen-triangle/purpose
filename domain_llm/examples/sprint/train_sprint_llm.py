#!/usr/bin/env python3
"""
Train a domain-specific language model on sprint data.

This example script demonstrates how to use the Domain-LLM framework
to train a model on sprint running data.
"""

import os
import logging
import argparse
from pathlib import Path

from domain_llm.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sprint_llm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sprint-llm-trainer")

# Constants
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "sprint_model"
DEFAULT_BASE_MODEL = "gpt2"  # Smaller model for faster training

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a sprint-specific language model")
    
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory with processed data")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Directory to save the trained model")
    parser.add_argument("--model-name", default=DEFAULT_BASE_MODEL,
                        help="Base model to fine-tune")
    parser.add_argument("--output-name", default=DEFAULT_MODEL_NAME,
                        help="Name of the output model directory")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length for training")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps for learning rate scheduler")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--corpus-file", default="sprint_corpus.txt",
                        help="Name of the corpus file")
    
    return parser.parse_args()

def check_data_readiness(data_dir, corpus_file):
    """Check if processed data is ready for training."""
    corpus_path = os.path.join(data_dir, corpus_file)
    
    if not os.path.exists(corpus_path):
        logger.warning(f"Training corpus not found at {corpus_path}")
        logger.info("You need to process the content data first.")
        logger.info("Run: python domain_llm/examples/sprint/process_content_data.py")
        return False
    
    logger.info(f"Found training corpus at {corpus_path}")
    return True

def main():
    """Main function to train the language model."""
    args = parse_args()
    
    logger.info("Starting sprint LLM training")
    
    # Check if data is ready
    if not check_data_readiness(args.data_dir, args.corpus_file):
        return
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        model_name=args.model_name,
        model_output_name=args.output_name,
        use_lora=args.use_lora,
        device="cpu" if args.force_cpu else None
    )
    
    # Load the data
    trainer.load_data(corpus_file=args.corpus_file)
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length
    )
    
    output_path = os.path.join(args.model_dir, args.output_name)
    logger.info(f"Training completed successfully. Model saved to {output_path}")
    logger.info(f"You can now use your model with: domain-llm generate --model-dir {output_path} --interactive --qa-mode")

if __name__ == "__main__":
    main() 