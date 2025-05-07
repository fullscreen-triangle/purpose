"""
Data Processing Module

This module handles data processing for the Purpose project,
including distributed processing of visualization data.
"""

import logging
from pathlib import Path
from typing import Optional
import psutil

from main.cli.distributed_processor import DistributedProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_data(
    data_dir: Path,
    output_dir: Path,
    max_examples: Optional[int] = None,
    memory_fraction: float = 0.4,
    chunk_size_mb: int = 50
):
    """
    Process data using distributed computing.
    
    Args:
        data_dir: Directory containing input data
        output_dir: Directory to save processed data
        max_examples: Maximum number of examples to process (optional)
        memory_fraction: Fraction of system memory to use (default: 0.4)
        chunk_size_mb: Size of each processing chunk in MB (default: 50)
    """
    try:
        # Check memory settings
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        requested_memory_gb = total_memory_gb * memory_fraction
        
        if memory_fraction > 0.4:
            logger.warning(
                f"Warning: Using {memory_fraction*100}% of system memory "
                f"({requested_memory_gb:.1f}GB). This may impact system performance. "
                "Consider using a lower value (0.4 or less recommended)."
            )
        
        # Use the content/visualisation directory where files actually exist
        visualisation_dir = Path("content") / "visualisation"
        if not visualisation_dir.exists():
            logger.error(f"Visualization directory not found at {visualisation_dir}")
            return
            
        logger.info(f"Using visualization files from {visualisation_dir}")
        
        # Initialize processor
        processor = DistributedProcessor(
            input_dir=visualisation_dir,  # Changed from data_dir / "visualization" / "content"
            output_dir=output_dir / "visualization",
            memory_fraction=memory_fraction,
            chunk_size_mb=chunk_size_mb
        )
        
        # Process visualization data
        logger.info(
            f"Starting distributed processing using {memory_fraction*100}% "
            f"of system memory ({requested_memory_gb:.1f}GB) "
            f"and {chunk_size_mb}MB chunks..."
        )
        processor.process_visualization_data()
        
        logger.info("Data processing complete!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise 