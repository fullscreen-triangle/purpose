#!/usr/bin/env python3
"""
Process and unify data from the content directory to create a training corpus for LLM.
This script is an example implementation that uses the Domain-LLM framework
to process sprint-specific data.
"""

import os
import logging
from pathlib import Path
from purpose.processor import CombinedDataProcessor
from purpose.examples.sprint.sprint_processor import SprintDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sprint_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sprint-example")

# Constants
CONTENT_DIR = "content"
OUTPUT_DIR = "data/processed"
OUTPUT_CORPUS = os.path.join(OUTPUT_DIR, "sprint_corpus.txt")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "sprint_data.jsonl")

def main():
    """Main function to process all data and create training corpus."""
    logger.info("Starting sprint data processing")
    
    # Create the processor
    processor = SprintDataProcessor(
        data_dir=CONTENT_DIR,
        output_dir=OUTPUT_DIR,
        output_corpus_filename="sprint_corpus.txt",
        output_jsonl_filename="sprint_data.jsonl"
    )
    
    # Process all files
    processor.process()
    
    logger.info(f"Processing complete. Created {OUTPUT_CORPUS} and {OUTPUT_JSONL}")
    logger.info("You can now train a model with: domain-llm train --data-dir data/processed --model-name gpt2 --output-name sprint_model")
    
if __name__ == "__main__":
    main() 