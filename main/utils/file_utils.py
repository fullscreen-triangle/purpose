import json
import os
import logging
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def save_json(data: Union[Dict, List], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data (Union[Dict, List]): Data to save
        filepath (str): Path to save the file
    """
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved data to {filepath}")

def load_json(filepath: str) -> Union[Dict, List]:
    """
    Load data from JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        Union[Dict, List]: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data from {filepath}")
    return data

def append_to_txt(text: str, filepath: str) -> None:
    """
    Append text to a file.
    
    Args:
        text (str): Text to append
        filepath (str): Path to the file
    """
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(text + '\n') 