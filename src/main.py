#!/usr/bin/env python3
"""
Main CLI entry point for the Sprint Results LLM Training Project.

This module provides a command-line interface for:
1. Scraping sprint results from World Athletics
2. Processing the scraped data
3. Training a domain-specific language model
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sprint_main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sprint-main")

def setup_scraper_command(subparsers):
    """Set up the 'scrape' command and its subcommands."""
    scrape_parser = subparsers.add_parser('scrape', help='Scrape sprint results data')
    scrape_parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    scrape_parser.add_argument('--output-dir', default='data/sprint_data', help='Directory to save scraped data')
    scrape_parser.set_defaults(func=run_scraper)

def setup_train_command(subparsers):
    """Set up the 'train' command and its subcommands."""
    train_parser = subparsers.add_parser('train', help='Train a domain-specific language model')
    
    # Add train-model subcommand
    train_model_parser = train_parser.add_subparsers(dest='train_command')
    model_parser = train_model_parser.add_parser('train-model', help='Train a language model on sprint data')
    
    # Add arguments for train-model
    model_parser.add_argument('--data-dir', default='data/processed', help='Directory with processed data')
    model_parser.add_argument('--model-dir', default='models', help='Directory to save trained models')
    model_parser.add_argument('--model-name', default='distilbert-base-uncased', help='Base model to fine-tune')
    model_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    model_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    model_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    model_parser.add_argument('--use-lora', action='store_true', help='Use LoRA for parameter-efficient fine-tuning')
    
    train_parser.set_defaults(func=run_train)

def run_scraper(args):
    """Run the sprint results scraper."""
    try:
        # Import here to avoid loading unnecessary modules
        from scraper.sprint_results_scraper import SprintResultsScraper
        
        logger.info(f"Starting sprint results scraper with output to {args.output_dir}")
        scraper = SprintResultsScraper(headless=args.headless, output_dir=args.output_dir)
        
        # Run the scraping process
        sprint_data = scraper.scrape_sprint_events()
        
        if sprint_data is not None:
            # Convert to text corpus for LLM training
            logger.info("Converting results to text corpus...")
            corpus_file = scraper.convert_to_text_corpus(sprint_data)
            logger.info(f"Text corpus created at: {corpus_file}")
            logger.info("You can now use this corpus for LLM training with the command:")
            logger.info("python -m src.main train train-model --data-dir data/processed --model-dir models")
            return 0
        else:
            logger.error("No sprint data was collected. Please check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")
        return 1

def run_train(args):
    """Run the model training process."""
    if not hasattr(args, 'train_command') or args.train_command is None:
        logger.error("No training subcommand specified. Use 'train-model'.")
        return 1
    
    if args.train_command == 'train-model':
        try:
            # Import the training module
            from training.model_trainer import train_model
            
            logger.info(f"Training model with data from {args.data_dir}")
            logger.info(f"Model will be saved to {args.model_dir}")
            logger.info(f"Using base model: {args.model_name}")
            logger.info(f"Training parameters: batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.epochs}")
            
            # Call the training function
            train_model(
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                use_lora=args.use_lora,
            )
            
            logger.info("Training completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return 1
    else:
        logger.error(f"Unknown training subcommand: {args.train_command}")
        return 1

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Sprint Results LLM Training Project')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Set up command parsers
    setup_scraper_command(subparsers)
    setup_train_command(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate function
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 