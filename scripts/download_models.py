#!/usr/bin/env python3
"""
Download Hugging Face Models

This script downloads models from Hugging Face for local use to avoid API issues.
Features:
- Retry mechanism with exponential backoff
- Direct HTTPS fallback download when API fails
- Local cache validation and management
"""

import os
import sys
import time
import argparse
import requests
import shutil
import json
import logging
from pathlib import Path
import random
from urllib.parse import urlparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up constants
MAX_RETRIES = 7
BASE_DELAY = 2  # seconds
HF_HUB_URL = "https://huggingface.co"
CACHE_INDEX_FILE = "cache_index.json"
HTTP_TIMEOUT = 300  # Increased HTTP timeout to 5 minutes

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not available. Will use fallback HTTP download method.")
    TRANSFORMERS_AVAILABLE = False

def load_cache_index(cache_dir):
    """Load the cache index if it exists"""
    index_path = os.path.join(cache_dir, CACHE_INDEX_FILE)
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache index: {e}")
    return {}

def save_cache_index(cache_dir, cache_index):
    """Save the cache index"""
    index_path = os.path.join(cache_dir, CACHE_INDEX_FILE)
    try:
        with open(index_path, 'w') as f:
            json.dump(cache_index, f, indent=2)
    except Exception as e:
        logger.warning(f"Error saving cache index: {e}")

def download_with_retry(model_id, output_dir, model_type="causal_lm", use_transformer_api=True):
    """
    Download a model with retry logic
    
    Args:
        model_id (str): Hugging Face model ID
        output_dir (str): Directory to save the model
        model_type (str): Type of model to download
        use_transformer_api (bool): Whether to try the Transformers API first
    """
    model_output_dir = os.path.join(output_dir, model_id.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize cache
    cache_dir = os.path.join(output_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_index = load_cache_index(cache_dir)
    
    # Check if model is already in cache
    if model_id in cache_index and os.path.exists(model_output_dir):
        logger.info(f"Model {model_id} found in cache, validating...")
        # Basic validation - could be enhanced with file checksums
        if all(os.path.exists(os.path.join(model_output_dir, file)) 
               for file in ["config.json", "pytorch_model.bin"]):
            logger.info(f"Using cached model {model_id}")
            return model_output_dir
    
    # Try different methods with retries
    methods_to_try = []
    
    if TRANSFORMERS_AVAILABLE and use_transformer_api:
        methods_to_try.append(("transformers_api", download_with_transformers))
    
    methods_to_try.append(("direct_https", download_with_https))
    
    for method_name, download_method in methods_to_try:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                logger.info(f"Attempting download with {method_name} (attempt {retries+1}/{MAX_RETRIES})")
                download_method(model_id, model_output_dir, model_type)
                
                # Update cache index
                cache_index[model_id] = {
                    "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method_name,
                    "path": model_output_dir
                }
                save_cache_index(cache_dir, cache_index)
                
                logger.info(f"Successfully downloaded {model_id} using {method_name}")
                return model_output_dir
            
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    logger.error(f"All {method_name} attempts failed for {model_id}. Error: {str(e)}")
                    break
                
                # Exponential backoff with jitter
                delay = BASE_DELAY * (2 ** retries) + random.uniform(0, 1)
                logger.warning(f"{method_name} attempt {retries} failed. Retrying in {delay:.2f}s. Error: {str(e)}")
                time.sleep(delay)
    
    raise Exception(f"Failed to download {model_id} after trying all methods")

def download_with_transformers(model_id, output_dir, model_type):
    """Download using transformers library"""
    # Download the tokenizer
    logger.info(f"Downloading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    
    # Download the model based on type
    logger.info(f"Downloading model for {model_id}...")
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)
    
    model.save_pretrained(output_dir)

def download_with_https(model_id, output_dir, model_type):
    """
    Download model files directly via HTTPS
    This is a simplified version - production code would need to handle
    specific model structures and file requirements
    """
    # Essential files to download for a basic model
    files_to_download = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt"
    ]
    
    parsed_model_id = model_id.split('/')
    if len(parsed_model_id) == 1:
        # Default namespace is the main HF organization
        namespace, model_name = "models", parsed_model_id[0]
    else:
        namespace, model_name = parsed_model_id
    
    for file in files_to_download:
        url = f"{HF_HUB_URL}/{namespace}/{model_name}/resolve/main/{file}"
        local_path = os.path.join(output_dir, file)
        
        try:
            logger.info(f"Downloading {url} to {local_path}")
            response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            elif response.status_code == 404:
                logger.warning(f"File {file} not found for model {model_id}")
                # Continue with other files, some models might not have all files
                continue
            else:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
            raise

def download_model(model_id, output_dir, model_type="causal_lm"):
    """
    Download a model and tokenizer from Hugging Face Hub to local directory.
    This function now uses the retry-enabled downloader.
    
    Args:
        model_id (str): Hugging Face model ID
        output_dir (str): Directory to save the model
        model_type (str): Type of model to download (causal_lm, base, etc.)
    """
    return download_with_retry(model_id, output_dir, model_type)

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for local use")
    
    parser.add_argument(
        "--model-id",
        type=str,
        help="Hugging Face model ID to download (e.g., 'distilgpt2')"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save downloaded models"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="causal_lm",
        choices=["causal_lm", "base", "sentence_transformer"],
        help="Type of model to download"
    )
    
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all common models used in the project"
    )
    
    parser.add_argument(
        "--force-https",
        action="store_true",
        help="Force using direct HTTPS download instead of transformers API"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the download cache before proceeding"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle cache clearing if requested
    if args.clear_cache:
        cache_dir = os.path.join(args.output_dir, "_cache")
        if os.path.exists(cache_dir):
            logger.info("Clearing download cache...")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
    
    if args.download_all:
        # List of commonly used models with their specific configurations
        common_models = [
            {"id": "distilgpt2", "type": "causal_lm"},
            {"id": "microsoft/phi-3-mini-4k-instruct", "type": "causal_lm"},
            {"id": "google/gemma-2-2b-it", "type": "causal_lm"},
            {"id": "intfloat/e5-large-v2", "type": "base"}  # This is a sentence transformer model
        ]
        
        logger.info(f"Starting individual downloads for {len(common_models)} models...")
        success_count = 0
        failed_models = []
        
        # Download each model individually
        for model_config in common_models:
            model_id = model_config["id"]
            model_type = model_config.get("type", args.model_type)
            
            logger.info(f"==== Starting download for {model_id} (type: {model_type}) ====")
            try:
                download_model(model_id, args.output_dir, model_type)
                success_count += 1
                logger.info(f"==== Successfully downloaded {model_id} ====")
            except Exception as e:
                logger.error(f"==== Failed to download {model_id}: {str(e)} ====")
                failed_models.append(model_id)
        
        # Summarize results
        logger.info(f"Download summary: {success_count}/{len(common_models)} models successful")
        if failed_models:
            logger.error(f"Failed models: {', '.join(failed_models)}")
            logger.info("You can try downloading the failed models individually with --model-id")
    
    elif args.model_id:
        try:
            download_model(
                args.model_id, 
                args.output_dir, 
                args.model_type
            )
            logger.info(f"Download of {args.model_id} completed successfully!")
        except Exception as e:
            logger.error(f"Error downloading {args.model_id}: {str(e)}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

# Command examples for reference:
# Download all models: python scripts/download_models.py --download-all
# Download a specific model: python scripts/download_models.py --model-id "distilgpt2"
# Download a specific model with type: python scripts/download_models.py --model-id "intfloat/e5-large-v2" --model-type "base"
# Force HTTPS download: python scripts/download_models.py --model-id "distilgpt2" --force-https
# Clear cache: python scripts/download_models.py --clear-cache --download-all 