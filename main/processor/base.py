"""
Base classes for data processing in Purpose project.
"""

import os
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, NamedTuple
from dataclasses import dataclass
from tqdm import tqdm

from main.processor.text_processor import SprintTextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("purpose.log"),
    ]
)
logger = logging.getLogger("purpose")

@dataclass
class ProcessorConfig:
    """Configuration class for data processors."""
    data_dir: str
    output_dir: str = "data/processed"
    output_corpus_filename: str = "purpose_corpus.txt"
    output_jsonl_filename: str = "domain_data.jsonl"
    max_file_size_mb: float = 50.0
    
    def __post_init__(self):
        """Convert string paths to Path objects after initialization."""
        self.data_dir = Path(str(self.data_dir))
        self.output_dir = Path(self.output_dir)

class ProcessorOutput(NamedTuple):
    """Output data from processor execution."""
    corpus_path: Path
    jsonl_path: Path
    num_records: int
    output_dir: Path

class BaseProcessor(ABC):
    """
    Base class for processing domain-specific data for LLM training.
    
    This class provides the abstract interface and common functionality
    for converting domain data into a format suitable for LLM training.
    """
    
    def __init__(
        self,
        config: ProcessorConfig,
    ):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration for the processor
        """
        self.config = config
        self.output_corpus = self.config.output_dir / self.config.output_corpus_filename
        self.output_jsonl = self.config.output_dir / self.config.output_jsonl_filename
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize records storage
        self.records = []
        self.logger = logger
    
    def process(self) -> ProcessorOutput:
        """
        Process all data files and create training corpus.
        
        This is the main method to call for processing data.
        
        Returns:
            ProcessorOutput with paths to created files and processing statistics
        """
        self.logger.info(f"Starting data processing from {self.config.data_dir}")
        
        # Process data files and collect records
        self.records = self._process_directory()
        
        # Create text corpus
        if self.records:
            self._create_text_corpus()
            self._create_jsonl_file()
            
            self.logger.info(f"Processing complete. Created {self.output_corpus} and {self.output_jsonl}")
            
            return ProcessorOutput(
                corpus_path=self.output_corpus,
                jsonl_path=self.output_jsonl,
                num_records=len(self.records),
                output_dir=self.config.output_dir
            )
        else:
            self.logger.warning("No records found. No output files created.")
            return ProcessorOutput(
                corpus_path=self.output_corpus,
                jsonl_path=self.output_jsonl,
                num_records=0,
                output_dir=self.config.output_dir
            )
    
    def _process_directory(self) -> List[Dict[str, Any]]:
        """
        Process all files in the data directory.
        
        Returns:
            List of processed records
        """
        self.logger.info(f"Processing directory: {self.config.data_dir}")
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

class CombinedDataProcessor(BaseProcessor):
    """
    Combined processor that applies all available processors
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/processed",
        max_file_size_mb: float = 50.0
    ):
        """
        Initialize the combined processor.
        
        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save processed data
            max_file_size_mb: Maximum file size to process in MB
        """
        config = ProcessorConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            max_file_size_mb=max_file_size_mb
        )
        super().__init__(config)
        
        # Import specific processors here to avoid circular imports

        # Initialize specific processors
        self.text_processor = SprintTextProcessor(config)
    
    def _find_data_files(self) -> List[Path]:
        """
        Find all data files using the specific processors.
        
        Returns:
            List of paths to data files
        """
        # Combine data files from all processors
        text_files = self.text_processor._find_data_files()
        
        # Deduplicate files
        return list(set(text_files))
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a file using the appropriate processor based on extension.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of processed records
        """
        suffix = file_path.suffix.lower()
        
        # Route to the appropriate processor
        if suffix in ['.txt', '.md', '.csv']:
            return self.text_processor._process_file(file_path)
        else:
            self.logger.warning(f"No processor available for {suffix} files: {file_path}")
            return []
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a processed record to text format.
        
        Args:
            record: Dictionary with record information
            
        Returns:
            Formatted text representation
        """
        # Use the text processor for all records for now
        return self.text_processor._record_to_text(record) 