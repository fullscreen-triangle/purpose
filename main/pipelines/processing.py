"""
Processing pipelines for the Purpose project.

This module provides pre-configured pipelines for data processing operations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from main.pipelines.base import BasePipeline
from main.utils.config import ProcessorConfig
from main.utils.services import get_processor


def create_processing_pipeline(
    data_dir: str,
    output_dir: str = "data/processed",
    max_file_size_mb: float = 50.0
) -> BasePipeline:
    """
    Create a data processing pipeline.
    
    Args:
        data_dir: Directory containing the raw data files
        output_dir: Directory to save processed data
        max_file_size_mb: Maximum file size to process in MB
        
    Returns:
        Configured processing pipeline
    """
    pipeline = BasePipeline(name="data_processing")
    
    # Stage 1: Setup
    def setup_stage(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the processing environment."""
        config = ProcessorConfig(
            data_dir=input_data.get('data_dir', data_dir),
            output_dir=input_data.get('output_dir', output_dir),
            max_file_size_mb=input_data.get('max_file_size_mb', max_file_size_mb)
        )
        processor = get_processor()
        processor.config = config
        return {
            'config': config,
            'processor': processor
        }
    
    # Stage 2: Processing
    def process_stage(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the data."""
        processor = input_data['processor']
        output = processor.process()
        return {
            'output': output,
            'processor': processor
        }
    
    # Stage 3: Post-processing
    def post_process_stage(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform any post-processing tasks."""
        output = input_data['output']
        # Generate statistics
        stats = {
            'num_records': output.num_records,
            'output_dir': str(output.output_dir),
            'corpus_path': str(output.corpus_path),
            'jsonl_path': str(output.jsonl_path)
        }
        return {
            'output': output,
            'stats': stats
        }
    
    # Add stages to pipeline
    pipeline.add_function(setup_stage, name="setup")
    pipeline.add_function(process_stage, name="process")
    pipeline.add_function(post_process_stage, name="post_process")
    
    return pipeline

def run_processing_pipeline(
    data_dir: str,
    output_dir: str = "data/processed",
    max_file_size_mb: float = 50.0
) -> Dict[str, Any]:
    """
    Run the data processing pipeline.
    
    Args:
        data_dir: Directory containing the raw data files
        output_dir: Directory to save processed data
        max_file_size_mb: Maximum file size to process in MB
        
    Returns:
        Pipeline output
    """
    pipeline = create_processing_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        max_file_size_mb=max_file_size_mb
    )
    
    # Prepare initial input
    input_data = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'max_file_size_mb': max_file_size_mb
    }
    
    # Run the pipeline
    result = pipeline.run(input_data)
    
    return result 