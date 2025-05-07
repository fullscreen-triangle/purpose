#!/usr/bin/env python3
"""
Full Pipeline Script for Sprint Domain Expert LLM

This script demonstrates the entire pipeline using Domain-LLM:
1. Process sprint data
2. Train a domain-specific LLM
3. Test the model with a predefined question

Usage:
    python run_pipeline.py [--no-process] [--no-train] [--model-name MODEL]

Options:
    --no-process    Skip the data processing step
    --no-train      Skip the model training step
    --model-name    Base model name to use (default: gpt2)
"""

import os
import logging
import argparse
import time
from pathlib import Path

from purpose.examples.sprint.sprint_processor import SprintDataProcessor
from purpose.trainer import ModelTrainer
from purpose.inference import ModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sprint_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sprint-pipeline")

# Constants
CONTENT_DIR = "content"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
MODEL_OUTPUT_NAME = "sprint_model"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete Sprint LLM pipeline")
    parser.add_argument("--no-process", action="store_true", help="Skip the data processing step")
    parser.add_argument("--no-train", action="store_true", help="Skip the model training step")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Base model name to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs") 
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient training")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if GPU is available")
    return parser.parse_args()


def main():
    """Run the full pipeline."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    start_time = time.time()
    logger.info("Starting Sprint LLM pipeline")
    
    # Step 1: Process sprint data
    if not args.no_process:
        logger.info("Step 1: Processing sprint data")
        processor = SprintDataProcessor(
            data_dir=CONTENT_DIR,
            output_dir=PROCESSED_DIR
        )
        processor.process()
    else:
        logger.info("Step 1: Skipping data processing (--no-process flag set)")
    
    # Step 2: Train model (optional)
    model_dir = os.path.join(MODELS_DIR, MODEL_OUTPUT_NAME)
    if not args.no_train:
        logger.info("Step 2: Training domain-specific LLM")
        
        # Create the trainer
        trainer = ModelTrainer(
            data_dir=PROCESSED_DIR,
            output_dir=MODELS_DIR,
            model_name=args.model_name,
            model_output_name=MODEL_OUTPUT_NAME,
            use_lora=args.use_lora,
            device="cpu" if args.force_cpu else None
        )
        
        # Load the data and train the model
        trainer.load_data(corpus_file="sprint_corpus.txt")
        trainer.train(
            batch_size=args.batch_size,
            learning_rate=5e-5,
            num_epochs=args.epochs
        )
        
        logger.info(f"Model training complete")
    else:
        logger.info("Step 2: Skipping model training (--no-train flag set)")
    
    # Step 3: Test model with a question
    logger.info("Step 3: Testing model with a sample question")
    test_question = "What are the key biomechanical factors that affect sprint performance?"
    
    try:
        # Load model for inference
        model = ModelInference(
            model_dir=model_dir,
            device="cpu" if args.force_cpu else None
        )
        
        # Generate answer
        logger.info(f"Asking: '{test_question}'")
        answer = model.answer_question(
            question=test_question,
            temperature=0.7
        )
        
        logger.info(f"Model answer: {answer}")
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
    
    # Pipeline complete
    total_time = time.time() - start_time
    logger.info(f"Pipeline complete in {total_time:.2f} seconds")
    logger.info(f"You can now use your model with: domain-llm generate --model-dir {model_dir} --interactive --qa-mode")


if __name__ == "__main__":
    main() 