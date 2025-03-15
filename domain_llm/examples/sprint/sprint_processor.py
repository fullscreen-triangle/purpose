"""
Sprint-specific data processor implementation for Domain-LLM.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any

from domain_llm.processor import CombinedDataProcessor

class SprintDataProcessor(CombinedDataProcessor):
    """
    Processor for sprint running data.
    
    This extends the CombinedDataProcessor with sprint-specific configurations
    and customizations.
    """
    
    def __init__(
        self,
        data_dir: str = "content",
        output_dir: str = "data/processed",
        csv_patterns: List[str] = ["**/*.csv"],
        json_patterns: List[str] = ["**/*.json"],
        output_corpus_filename: str = "sprint_corpus.txt",
        output_jsonl_filename: str = "sprint_data.jsonl",
        max_file_size_mb: float = 50.0,
        **kwargs
    ):
        """
        Initialize the sprint data processor.
        
        Args:
            data_dir: Directory containing sprint data files
            output_dir: Directory to save processed data
            csv_patterns: Glob patterns to find CSV files
            json_patterns: Glob patterns to find JSON files
            output_corpus_filename: Filename for the output text corpus
            output_jsonl_filename: Filename for the output JSONL file
            max_file_size_mb: Maximum file size to process (in MB)
            **kwargs: Additional processor-specific arguments
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            csv_patterns=csv_patterns,
            json_patterns=json_patterns,
            max_file_size_mb=max_file_size_mb,
            output_corpus_filename=output_corpus_filename,
            output_jsonl_filename=output_jsonl_filename,
            **kwargs
        )
    
    def _is_race_results(self, columns: List[str]) -> bool:
        """
        Check if CSV columns indicate sprint race results.
        
        Args:
            columns: List of column names
            
        Returns:
            True if the columns match sprint race results
        """
        race_columns = ["pos", "RANK", "Athlete", "mark", "Country", "bib", "lane", "time", "result"]
        return any(col in columns for col in race_columns)
    
    def _is_sprint_data(self, file_path: Path) -> bool:
        """
        Check if a file contains sprint-related data.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is likely sprint-related
        """
        sprint_keywords = ["100m", "200m", "400m", "sprint", "race", "athlete", 
                         "bolt", "relay", "championship", "olympic", "track"]
        
        # Check filename
        if any(keyword in str(file_path).lower() for keyword in sprint_keywords):
            return True
        
        # Check directory name
        parts = str(file_path).split(os.sep)
        if any(keyword in parts[-2].lower() for keyword in sprint_keywords):
            return True
        
        return False
    
    def process(self) -> None:
        """
        Process all sprint data files and create training corpus.
        
        This overrides the base method to add sprint-specific filtering.
        """
        self.logger.info(f"Starting sprint data processing from {self.data_dir}")
        
        # Process CSV files
        csv_records = self.csv_processor._process_directory()
        
        # Process JSON files
        json_records = self.json_processor._process_directory()
        
        # Combine records
        self.records = csv_records + json_records
        
        # Create text corpus
        if self.records:
            self._create_text_corpus()
            self._create_jsonl_file()
            
            self.logger.info(f"Processing complete. Created {self.output_corpus} and {self.output_jsonl}")
            self.logger.info(f"Processed {len(self.records)} sprint-related records")
        else:
            self.logger.warning("No sprint records found. No output files created.")


def main():
    """Main function to demonstrate sprint data processing."""
    # Create processor
    processor = SprintDataProcessor(
        data_dir="content",
        output_dir="data/processed"
    )
    
    # Process data
    processor.process()
    
    print(f"Processing complete. Output files saved to {processor.output_dir}")

if __name__ == "__main__":
    main() 