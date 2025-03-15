"""
Base classes for data processing in Domain-LLM
"""

import os
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("domain_llm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("domain-llm")

class DataProcessor(ABC):
    """
    Base class for processing domain-specific data for LLM training.
    
    This class provides the abstract interface and common functionality
    for converting domain data into a format suitable for LLM training.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/processed",
        output_corpus_filename: str = "purpose_corpus.txt",
        output_jsonl_filename: str = "domain_data.jsonl",
        **kwargs
    ):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save processed data
            output_corpus_filename: Filename for the output text corpus
            output_jsonl_filename: Filename for the output JSONL file
            **kwargs: Additional processor-specific arguments
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_corpus = self.output_dir / output_corpus_filename
        self.output_jsonl = self.output_dir / output_jsonl_filename
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store additional configuration
        self.config = kwargs
        
        # Initialize records storage
        self.records = []
        self.logger = logger
    
    def process(self) -> None:
        """
        Process all data files and create training corpus.
        
        This is the main method to call for processing data.
        """
        self.logger.info(f"Starting data processing from {self.data_dir}")
        
        # Process data files and collect records
        self.records = self._process_directory()
        
        # Create text corpus
        if self.records:
            self._create_text_corpus()
            self._create_jsonl_file()
            
            self.logger.info(f"Processing complete. Created {self.output_corpus} and {self.output_jsonl}")
        else:
            self.logger.warning("No records found. No output files created.")
    
    def _process_directory(self) -> List[Dict[str, Any]]:
        """
        Process all files in the data directory.
        
        Returns:
            List of processed records
        """
        self.logger.info(f"Processing directory: {self.data_dir}")
        all_records = []
        
        # Find and process all data files
        data_files = self._find_data_files()
        
        # Process each file
        for file_path in tqdm(data_files, desc="Processing files"):
            try:
                records = self._process_file(file_path)
                all_records.extend(records)
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
        
        self.logger.info(f"Processed {len(all_records)} total records")
        return all_records
    
    @abstractmethod
    def _find_data_files(self) -> List[Path]:
        """
        Find all data files to process.
        
        Returns:
            List of paths to data files
        """
        pass
    
    @abstractmethod
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of processed records
        """
        pass
    
    @abstractmethod
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a processed record to human-readable text format.
        
        Args:
            record: Dictionary with record information
            
        Returns:
            Formatted text representation
        """
        pass
    
    def _create_text_corpus(self) -> Path:
        """
        Convert processed records to a text corpus for LLM training.
        
        Returns:
            Path to the saved corpus file
        """
        self.logger.info(f"Creating text corpus with {len(self.records)} records")
        
        # Convert each record to text and join with separators
        text_passages = []
        for record in tqdm(self.records, desc="Converting records to text"):
            passage = f"""
--- START OF DOCUMENT ---

{self._record_to_text(record)}

--- END OF DOCUMENT ---
"""
            text_passages.append(passage)
        
        # Write to output file
        with open(self.output_corpus, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_passages))
        
        self.logger.info(f"Created text corpus at {self.output_corpus}")
        return self.output_corpus
    
    def _create_jsonl_file(self) -> Path:
        """
        Save processed records as JSONL for easier processing.
        
        Returns:
            Path to the saved JSONL file
        """
        self.logger.info(f"Creating JSONL file with {len(self.records)} records")
        
        # Write each record as a JSON line
        with open(self.output_jsonl, 'w', encoding='utf-8') as f:
            for record in tqdm(self.records, desc="Writing JSONL file"):
                f.write(json.dumps(record) + '\n')
        
        self.logger.info(f"Created JSONL file at {self.output_jsonl}")
        return self.output_jsonl 