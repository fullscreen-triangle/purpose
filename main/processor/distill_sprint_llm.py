#!/usr/bin/env python3
"""
Sprint Model Knowledge Distillation Script

This script distills knowledge from large LLMs (OpenAI and Claude) into a smaller
domain-specific model for sprint running, using scientific papers as source material.

Usage:
    python distill_sprint_llm.py [--papers-dir PAPERS_DIR] [--output-dir OUTPUT_DIR] [--model-name MODEL_NAME]
"""

import os
import argparse
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path

from main.trainer.enhanced_distillation import run_enhanced_distillation
from main.trainer.knowledge_distillation import run_distillation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sprint_llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key:
        logger.warning("OPENAI_API_KEY not found in environment. Required for knowledge distillation.")
    
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment. Required for enhanced answers.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distill knowledge from LLMs into a smaller sprint running model")
    parser.add_argument("--papers-dir", type=str, default="papers", 
                        help="Directory containing PDF papers (default: papers)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save the model (default: models)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory for processed data (default: data/processed)")
    parser.add_argument("--model-name", type=str, default="distilgpt2",
                        help="Student model name (default: distilgpt2)")
    parser.add_argument("--use-llama", action="store_true",
                        help="Use a local LLaMA model instead of default models")
    parser.add_argument("--llama-path", type=str,
                        help="Path to local LLaMA model (required if use-llama is set)")
    parser.add_argument("--num-qa-pairs", type=int, default=100,
                        help="Number of QA pairs to generate (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--quantize", type=str, choices=["true", "false"], default="true",
                        help="Use quantization to reduce model size (default: true)")
    parser.add_argument("--bit-precision", type=str, choices=['4', '8', 'none'], default='8',
                        help="Quantization precision (4-bit, 8-bit, or none) (default: 8)")
    parser.add_argument("--lora-r", type=int, default=4,
                        help="LoRA attention dimension (default: 4, lower = smaller model)")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use enhanced distillation with multi-stage approach (default: false)")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Handle LLaMA models
    if args.use_llama:
        if not args.llama_path:
            logger.error("--llama-path is required when --use-llama is specified")
            sys.exit(1)
        
        # Override model_name with llama_path
        args.model_name = args.llama_path
        logger.info(f"Using local LLaMA model from: {args.llama_path}")
        
        # Check for required libraries
        try:
            import bitsandbytes
            from transformers import LlamaForCausalLM, LlamaTokenizer
            logger.info("Found required libraries for LLaMA models")
        except ImportError as e:
            logger.error(f"Missing required libraries for LLaMA models: {e}")
            logger.error("Please install: pip install bitsandbytes transformers>=4.30.0")
            sys.exit(1)
    
    # Handle quantization based on bit precision
    if args.bit_precision == '4':
        quantize = True
        logger.info("Using 4-bit quantization for reduced memory usage")
    elif args.bit_precision == '8' or args.quantize.lower() == "true":
        quantize = True
        logger.info("Using 8-bit quantization for reduced memory usage")
    else:
        quantize = False
        logger.info("Using full precision (no quantization)")
    
    # Run distillation
    if args.enhanced:
        logger.info("Starting enhanced knowledge distillation process...")
    else:
        logger.info("Starting knowledge distillation process...")
    
    try:
        # Choose the appropriate distillation function
        if args.enhanced:
            distillation_func = run_enhanced_distillation
            logger.info("Using enhanced multi-stage distillation approach:")
            logger.info("  - Stage 1: Creating knowledge map from papers")
            logger.info("  - Stage 2: Generating stratified queries across knowledge dimensions")
            logger.info("  - Stage 3: Producing enhanced responses with knowledge grounding")
            logger.info("  - Stage 4: Training with curriculum learning")
        else:
            distillation_func = run_distillation
        
        # Run the selected distillation process
        model_path = distillation_func(
            papers_dir=args.papers_dir,
            processed_data_dir=args.data_dir,
            models_dir=args.output_dir,
            student_model_name=args.model_name,
            num_qa_pairs=args.num_qa_pairs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            quantize=quantize,
            lora_r=args.lora_r
        )
        
        if model_path:
            if args.enhanced:
                logger.info(f"Enhanced knowledge distillation complete! Model saved to {model_path}")
            else:
                logger.info(f"Knowledge distillation complete! Model saved to {model_path}")
        else:
            logger.error("Knowledge distillation failed to produce a model. Check logs for details.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during knowledge distillation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 