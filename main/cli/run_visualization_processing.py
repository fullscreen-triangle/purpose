#!/usr/bin/env python3
"""
Visualization Processing Runner

This script runs the entire pipeline to process visualization packages
and build models. It's designed to be started and left running.

Usage:
    python -m purpose.cli.run_visualization_processing --data-dir /path/to/data --output-dir /path/to/output
"""

import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import traceback

from main.cli.process import process_data
from main.cli.runner import run_train
from main.examples.sprint.enhanced_training import run_enhanced_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization-pipeline")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Visualization Processing Pipeline")
    
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing input data")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Directory to save processed data and models")
    parser.add_argument("--memory-fraction", type=float, default=0.4,
                       help="Fraction of system memory to use (default: 0.4)")
    parser.add_argument("--chunk-size-mb", type=int, default=50,
                       help="Size of processing chunks in MB (default: 50)")
    parser.add_argument("--model-name", type=str, default="gpt2",
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip data processing and only train models")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training and only process data")
    parser.add_argument("--max-runtime", type=int, default=180,
                       help="Maximum runtime in minutes (default: 3 hours)")
    
    return parser.parse_args()

def parse_enhanced_args(args_list):
    """
    Parse arguments for the enhanced training pipeline.
    
    Args:
        args_list: List of command line arguments
        
    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description="Enhanced Training Pipeline Parameters")
    
    # Pipeline control arguments
    parser.add_argument("--run-all", action="store_true",
                       help="Run all steps of the enhanced pipeline")
    parser.add_argument("--extract-only", action="store_true",
                       help="Only run the extraction step")
    parser.add_argument("--map-only", action="store_true",
                       help="Only run the knowledge mapping step")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only run the QA generation step")
    parser.add_argument("--train-only", action="store_true",
                       help="Only run the model training step")
    
    # Model parameters
    parser.add_argument("--base-model", type=str, default="gpt2",
                       help="Base model to use for training")
    parser.add_argument("--output-model-name", type=str, default="enhanced_model",
                       help="Name for the output model")
    parser.add_argument("--training-steps", type=int, default=3,
                       help="Number of training epochs/steps")
    
    # Data parameters
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of QA pairs to generate")
    parser.add_argument("--max-source-length", type=int, default=1024,
                       help="Maximum length of source text")
    parser.add_argument("--max-target-length", type=int, default=512,
                       help="Maximum length of target text")
    
    # Advanced options
    parser.add_argument("--use-specialized-models", action="store_true",
                       help="Use domain-specialized models for each pipeline stage")
    parser.add_argument("--extraction-model", type=str, default=None,
                       help="Model to use for knowledge extraction")
    parser.add_argument("--mapping-model", type=str, default=None,
                       help="Model to use for knowledge mapping")
    parser.add_argument("--generation-model", type=str, default=None,
                       help="Model to use for QA generation")
    parser.add_argument("--use-curriculum", action="store_true",
                       help="Use curriculum learning for training")
    
    return parser.parse_args(args_list)

def main():
    """Run the visualization processing pipeline."""
    start_time = datetime.now()
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    viz_output_dir = output_dir / "visualization"
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate end time
    end_time = start_time + timedelta(minutes=args.max_runtime)
    
    logger.info(f"Starting visualization pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Will run until approximately {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using {args.memory_fraction*100}% of system memory")

    try:
        # Step 1: Process visualization data
        if not args.skip_processing:
            logger.info("=== STEP 1: Processing Visualization Data ===")
            process_data(
                data_dir=data_dir,
                output_dir=output_dir,
                memory_fraction=args.memory_fraction,
                chunk_size_mb=args.chunk_size_mb
            )
            logger.info("Visualization data processing complete!")
        else:
            logger.info("Skipping data processing as requested")
        
        # Step 2: Train visualization model
        if not args.skip_training and viz_output_dir.exists():
            logger.info("=== STEP 2: Training Visualization Model ===")
            
            # Training with enhanced pipeline for knowledge distillation
            # the enhanced parser function below does not exist
            enhanced_args = parse_enhanced_args([
                "--run-all",
                "--base-model", args.model_name,
                "--output-model-name", "visualization_model",
                "--training-steps", str(args.epochs),
                "--num-samples", "50"
            ])
            run_enhanced_pipeline(enhanced_args)
            
            # Additional training directly on visualization corpus
            run_train(
                data_dir=str(viz_output_dir),
                model_dir=str(output_dir / "models"),
                model_name=args.model_name,
                output_name="visualization_specialized_model",
                epochs=args.epochs,
                use_lora=True,
                memory_fraction=args.memory_fraction
            )
            
            logger.info("Model training complete!")
        elif args.skip_training:
            logger.info("Skipping model training as requested")
        else:
            logger.warning("Visualization output directory not found, skipping training")
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up and report total runtime
        elapsed_time = datetime.now() - start_time
        logger.info(f"Total runtime: {elapsed_time}")

if __name__ == "__main__":
    main() 