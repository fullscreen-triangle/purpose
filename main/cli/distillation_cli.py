#!/usr/bin/env python3
"""
Distillation CLI Module

This module provides a clean interface for running the enhanced distillation pipeline,
properly integrating the CLI commands with the trainer's implementation.
"""

import os
import logging
import sys
from pathlib import Path
import random
import numpy as np
import torch

# Apply compatibility patches before importing transformers
try:
    from main.utils.transformers_patch import apply_patches
    # Explicitly call the apply_patches function
    apply_patches()
    logging.info("Applied transformers compatibility patches")
except ImportError as e:
    print(f"Warning: Could not apply transformers compatibility patches: {e}")
except Exception as e:
    print(f"Error applying transformers patches: {e}")

try:
    from transformers import set_seed
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please run setup_env.py to install compatible versions of required packages")
    sys.exit(1)

try:
    from main.trainer.enhanced_distillation import run_enhanced_distillation
except ImportError as e:
    print(f"Error importing enhanced_distillation module: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("distillation_cli.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("distillation-cli")

def run_distillation_pipeline(
    papers_dir: str,
    data_dir: str = None,
    output_dir: str = None,
    model_name: str = "distilgpt2",
    num_qa_pairs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    epochs: int = 3,
    quantize: bool = False,
    lora_r: int = 4,
    seed: int = 42
) -> str:
    """
    Main function to run the distillation pipeline.
    
    Args:
        papers_dir: Directory containing PDF papers
        data_dir: Directory for processed data
        output_dir: Directory to save models
        model_name: Base model to fine-tune
        num_qa_pairs: Number of QA pairs to generate
        batch_size: Training batch size
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        quantize: Whether to use quantization
        lora_r: LoRA attention dimension
        seed: Random seed for reproducibility
        
    Returns:
        Path to the trained model
    """
    # Set random seeds for reproducibility
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create default directories if not provided
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data", "processed")
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "models")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting distillation pipeline with model: {model_name}")
    logger.info(f"Papers directory: {papers_dir}")
    logger.info(f"Processed data directory: {data_dir}")
    logger.info(f"Models directory: {output_dir}")
    
    # Run the enhanced distillation pipeline
    try:
        model_path = run_enhanced_distillation(
            papers_dir=papers_dir,
            processed_data_dir=data_dir,
            models_dir=output_dir,
            student_model_name=model_name,
            num_qa_pairs=num_qa_pairs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            quantize=quantize,
            lora_r=lora_r
        )
        
        if model_path:
            logger.info(f"Distillation completed successfully. Model saved to: {model_path}")
            return model_path
        else:
            logger.error("Distillation failed. Check logs for details.")
            return None
    except Exception as e:
        logger.error(f"Error in distillation pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced knowledge distillation pipeline")
    
    parser.add_argument(
        "--papers-dir",
        type='.././papers/',
        required=True,
        help="Directory containing PDF papers"
    )
    
    parser.add_argument(
        "--data-dir",
        type='.././data/processed/',
        help="Directory for processed data"
    )
    
    parser.add_argument(
        "--output-dir",
        type='.././output/',
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilgpt2",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--num-qa-pairs",
        type=int,
        default=100,
        help="Number of QA pairs to generate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantization to reduce model size"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=4,
        help="LoRA attention dimension"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    run_distillation_pipeline(
        papers_dir=args.papers_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_qa_pairs=args.num_qa_pairs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        quantize=args.quantize,
        lora_r=args.lora_r,
        seed=args.seed
    ) 