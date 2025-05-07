#!/usr/bin/env python3
"""
Enhanced Distillation Runner

This script provides a simple entry point to run the enhanced distillation pipeline.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Apply compatibility patches before importing transformers
try:
    from main.utils.transformers_patch import apply_patches
    apply_patches()
except ImportError:
    print("Warning: Could not apply transformers compatibility patches.")

# Import the distillation pipeline
from main.cli.distillation_cli import run_distillation_pipeline

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced knowledge distillation pipeline")
    
    parser.add_argument(
        "--papers-dir",
        type=str,
        required=True,
        help="Directory containing PDF papers"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory for processed data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
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
    
    # Run the distillation pipeline
    model_path = run_distillation_pipeline(
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
    
    # Print success or failure message
    if model_path:
        print(f"\nDistillation completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"\nYou can use this model with the 'purpose generate' command:")
        print(f"  purpose generate --model-dir {model_path} --prompt \"Your prompt here\"")
    else:
        print("\nDistillation failed. Check the logs for details.")
        sys.exit(1) 