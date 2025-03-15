"""
Implementations of data processors for common file formats.
"""

import os
import glob
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from purpose.processor.base import DataProcessor

class CSVDataProcessor(DataProcessor):
    """
    Processor for CSV data files.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/processed",
        patterns: List[str] = ["**/*.csv"],
        **kwargs
    ):
        """
        Initialize the CSV data processor.
        
        Args:
            data_dir: Directory containing CSV files
            output_dir: Directory to save processed data
            patterns: Glob patterns to find CSV files
            **kwargs: Additional processor-specific arguments
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            output_corpus_filename="csv_corpus.txt",
            output_jsonl_filename="csv_data.jsonl",
            **kwargs
        )
        self.patterns = patterns
    
    def _find_data_files(self) -> List[Path]:
        """
        Find all CSV files matching the patterns.
        
        Returns:
            List of paths to CSV files
        """
        all_files = []
        for pattern in self.patterns:
            files = glob.glob(str(self.data_dir / pattern), recursive=True)
            all_files.extend([Path(f) for f in files])
        
        self.logger.info(f"Found {len(all_files)} CSV files")
        return all_files
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of processed records
        """
        self.logger.info(f"Processing CSV file: {file_path}")
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Extract context from file path
            parts = str(file_path).split(os.sep)
            source = parts[-2] if len(parts) > 1 else "Unknown"
            
            # Process each row
            records = []
            for _, row in df.iterrows():
                record = {
                    "source": source,
                    "file": file_path.name
                }
                
                # Add all available fields
                for col in df.columns:
                    if pd.notna(row[col]):
                        record[col] = str(row[col])
                
                records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            return []
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a record to text format.
        
        Args:
            record: Dictionary with record information
            
        Returns:
            Formatted text representation
        """
        # Start with record source information
        source = record.get('source', 'Unknown')
        filename = record.get('file', 'Unknown')
        
        # Create heading
        text = [f"SOURCE: {source}"]
        text.append(f"FILE: {filename}")
        
        # Add all fields
        for key, value in sorted(record.items()):
            if key not in ['source', 'file'] and value is not None:
                text.append(f"{key.upper()}: {value}")
        
        return "\n".join(text)


class JSONDataProcessor(DataProcessor):
    """
    Processor for JSON data files.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/processed",
        patterns: List[str] = ["**/*.json"],
        max_file_size_mb: float = 50.0,
        **kwargs
    ):
        """
        Initialize the JSON data processor.
        
        Args:
            data_dir: Directory containing JSON files
            output_dir: Directory to save processed data
            patterns: Glob patterns to find JSON files
            max_file_size_mb: Maximum file size to process (in MB)
            **kwargs: Additional processor-specific arguments
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            output_corpus_filename="json_corpus.txt",
            output_jsonl_filename="json_data.jsonl",
            **kwargs
        )
        self.patterns = patterns
        self.max_file_size_mb = max_file_size_mb
    
    def _find_data_files(self) -> List[Path]:
        """
        Find all JSON files matching the patterns.
        
        Returns:
            List of paths to JSON files
        """
        all_files = []
        for pattern in self.patterns:
            files = glob.glob(str(self.data_dir / pattern), recursive=True)
            all_files.extend([Path(f) for f in files])
        
        self.logger.info(f"Found {len(all_files)} JSON files")
        return all_files
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of processed records
        """
        self.logger.info(f"Processing JSON file: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > self.max_file_size_mb:
            self.logger.warning(f"Skipping large file {file_path} ({file_size:.1f} MB)")
            return []
        
        try:
            # Read JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract context from file path
            parts = str(file_path).split(os.sep)
            source = parts[-2] if len(parts) > 1 else "Unknown"
            data_type = file_path.name.replace('.json', '')
            
            # Handle different data structures
            if isinstance(data, list):
                # List of records
                records = []
                for item in data:
                    if isinstance(item, dict):
                        item['source'] = source
                        item['data_type'] = data_type
                        records.append(item)
                return records
                
            elif isinstance(data, dict):
                # Single record or nested structure
                if "data" in data and isinstance(data["data"], list):
                    # Extract and enrich list data
                    records = []
                    for item in data["data"]:
                        if isinstance(item, dict):
                            item['source'] = source
                            item['data_type'] = data_type
                            # Add top-level metadata if available
                            for key, value in data.items():
                                if key != "data" and not isinstance(value, (list, dict)):
                                    item[f"meta_{key}"] = value
                            records.append(item)
                    return records
                else:
                    # Process as a single record with nested information
                    # Flatten complex nested structures for better text representation
                    flat_records = self._flatten_json(data, source, data_type)
                    return flat_records
            else:
                self.logger.warning(f"Unexpected JSON structure in {file_path}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            return []
    
    def _flatten_json(
        self, 
        data: Any, 
        source: str, 
        data_type: str, 
        parent_key: str = '', 
        records: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively flatten a JSON structure into a list of records.
        
        Args:
            data: The JSON data to flatten
            source: Source directory of the file
            data_type: Type of data derived from filename
            parent_key: Parent key for nested structures
            records: Accumulator for records
            
        Returns:
            List of flattened records
        """
        if records is None:
            records = []
        
        # Base case: data is a simple value
        if not isinstance(data, (dict, list)):
            return [{"key": parent_key, "value": data, "source": source, "data_type": data_type}]
        
        # Process dictionary
        if isinstance(data, dict):
            # Check if this is a record-like dictionary
            if any(key in data for key in ["name", "id", "athlete", "time", "position", "mark"]):
                # Looks like a record, add source information
                record = data.copy()
                record["source"] = source
                record["data_type"] = data_type
                records.append(record)
            else:
                # Process each key in the dictionary
                for key, value in data.items():
                    new_key = f"{parent_key}.{key}" if parent_key else key
                    
                    if isinstance(value, (dict, list)):
                        # Recursively process nested structures
                        self._flatten_json(value, source, data_type, new_key, records)
                    else:
                        # Add simple key-value pair
                        records.append({
                            "key": new_key, 
                            "value": value, 
                            "source": source, 
                            "data_type": data_type
                        })
        
        # Process list
        elif isinstance(data, list):
            # Check if list contains record-like dictionaries
            if data and isinstance(data[0], dict) and any(key in data[0] for key in ["name", "id", "athlete", "time", "position", "mark"]):
                # Add source information to each record
                for item in data:
                    if isinstance(item, dict):
                        record = item.copy()
                        record["source"] = source
                        record["data_type"] = data_type
                        records.append(record)
            else:
                # Process each item in the list
                for i, item in enumerate(data):
                    new_key = f"{parent_key}[{i}]" if parent_key else f"item_{i}"
                    
                    if isinstance(item, (dict, list)):
                        # Recursively process nested structures
                        self._flatten_json(item, source, data_type, new_key, records)
                    else:
                        # Add simple list item
                        records.append({
                            "key": new_key, 
                            "value": item, 
                            "source": source, 
                            "data_type": data_type
                        })
        
        return records
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a record to text format.
        
        Args:
            record: Dictionary with record information
            
        Returns:
            Formatted text representation
        """
        # Start with record source information
        source = record.get('source', 'Unknown')
        data_type = record.get('data_type', 'Unknown')
        
        # Create heading
        text = [f"SOURCE: {source}"]
        text.append(f"TYPE: {data_type}")
        
        # Handle key-value records differently
        if "key" in record and "value" in record:
            text.append(f"{record['key'].upper()}: {record['value']}")
        else:
            # Add all fields
            for key, value in sorted(record.items()):
                if key not in ['source', 'data_type'] and value is not None:
                    text.append(f"{key.upper()}: {value}")
        
        return "\n".join(text)


class CombinedDataProcessor(DataProcessor):
    """
    Process both CSV and JSON files in a data directory.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "data/processed",
        csv_patterns: List[str] = ["**/*.csv"],
        json_patterns: List[str] = ["**/*.json"],
        max_file_size_mb: float = 50.0,
        **kwargs
    ):
        """
        Initialize the combined data processor.
        
        Args:
            data_dir: Directory containing data files
            output_dir: Directory to save processed data
            csv_patterns: Glob patterns to find CSV files
            json_patterns: Glob patterns to find JSON files
            max_file_size_mb: Maximum file size to process (in MB)
            **kwargs: Additional processor-specific arguments
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            **kwargs
        )
        
        # Create individual processors
        self.csv_processor = CSVDataProcessor(
            data_dir=data_dir,
            output_dir=output_dir,
            patterns=csv_patterns
        )
        
        self.json_processor = JSONDataProcessor(
            data_dir=data_dir,
            output_dir=output_dir,
            patterns=json_patterns,
            max_file_size_mb=max_file_size_mb
        )
    
    def _find_data_files(self) -> List[Path]:
        """
        Find all CSV and JSON files.
        
        Note: This method is not actually used in the combined processor.
        
        Returns:
            Empty list (placeholder)
        """
        return []
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a data file.
        
        Note: This method is not actually used in the combined processor.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Empty list (placeholder)
        """
        return []
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """
        Convert a record to text format.
        
        Args:
            record: Dictionary with record information
            
        Returns:
            Formatted text representation
        """
        # Determine which processor to use based on the file extension
        if "file" in record and record["file"].endswith(".csv"):
            return self.csv_processor._record_to_text(record)
        else:
            return self.json_processor._record_to_text(record)
    
    def process(self) -> None:
        """
        Process all data files and create training corpus.
        
        This method overrides the base class method to use both processors.
        """
        self.logger.info(f"Starting combined data processing from {self.data_dir}")
        
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
        else:
            self.logger.warning("No records found. No output files created.") 