#!/usr/bin/env python3
"""
Codespace Visualization Processor

This is a simplified version of the visualization processor designed to run
efficiently in GitHub Codespaces with improved stability and error handling.
"""

import os
import json
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import gc
import re
import concurrent.futures
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("codespace_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingState:
    """Tracks processing state and supports resuming."""
    
    def __init__(self, state_dir: Path):
        """Initialize processing state."""
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = state_dir / "processing_state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load processing state from disk."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "processed_files": [],
            "current_file": None,
            "total_processed": 0,
            "last_update": None
        }
    
    def save_state(self):
        """Save current processing state to disk."""
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def mark_file_complete(self, file_path: str, processed_items: int):
        """Mark a file as completely processed."""
        self.state["processed_files"].append(file_path)
        self.state["current_file"] = None
        self.state["total_processed"] += processed_items
        self.save_state()
    
    def is_file_processed(self, file_path: str) -> bool:
        """Check if a file has been processed."""
        return file_path in self.state["processed_files"]
    
    def set_current_file(self, file_path: str):
        """Set the current file being processed."""
        self.state["current_file"] = file_path
        self.save_state()

class CodespaceProcessor:
    """Processes visualization data in GitHub Codespaces."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        temp_dir: Optional[Path] = None,
        max_workers: int = 2,
        batch_size: int = 50
    ):
        """Initialize codespace processor."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="codespace_processing_"))
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize state
        self.state = ProcessingState(self.temp_dir)
    
    def _detect_visualization_type(self, code: str) -> str:
        """Detect the type of visualization from the code."""
        code_lower = code.lower()
        
        if "d3" in code_lower:
            if "force" in code_lower:
                return "force_directed"
            elif "tree" in code_lower:
                return "tree"
            elif "map" in code_lower:
                return "map"
            elif "bar" in code_lower:
                return "bar_chart"
            elif "line" in code_lower:
                return "line_chart"
            elif "scatter" in code_lower:
                return "scatter_plot"
            return "d3_other"
            
        if "react" in code_lower:
            return "react"
            
        if "chart.js" in code_lower or "chartjs" in code_lower:
            return "chartjs"
            
        if "leaflet" in code_lower:
            return "leaflet"
            
        if "plotly" in code_lower:
            return "plotly"
            
        return "other"
    
    def _extract_metadata(self, file_path: str, code: str) -> Dict[str, Any]:
        """Extract metadata from the code."""
        file_type = Path(file_path).suffix.lower()
        
        # Extract libraries
        libraries = []
        import_pattern = r'import\s+([^;]+?)\s+from\s+[\'"]([^\'"]+)[\'"]'
        require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
        
        for match in re.finditer(import_pattern, code):
            libraries.append(match.group(2))
        
        for match in re.finditer(require_pattern, code):
            libraries.append(match.group(1))
        
        # Check for styles
        has_styles = "style" in code.lower() or "css" in code.lower()
        
        return {
            "file_type": file_type,
            "size": len(code),
            "libraries": list(set(libraries)),
            "has_styles": has_styles
        }
    
    def _extract_code(self, code: str) -> str:
        """Clean the code by removing comments and unnecessary whitespace."""
        # Remove single-line comments
        code = re.sub(r'\/\/.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)
        
        # Remove empty lines and extra whitespace
        code_lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line:
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _process_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Process a single visualization file."""
        try:
            # Clean code
            cleaned_code = self._extract_code(content)
            
            # Detect visualization type
            viz_type = self._detect_visualization_type(cleaned_code)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, cleaned_code)
            
            return {
                "file_path": file_path,
                "type": viz_type,
                "code": cleaned_code,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of files."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_file, item["file_path"], item["content"])
                for item in batch
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")
        
        return results
    
    def _extract_files_from_zip(self, zip_path: Path) -> List[Dict[str, str]]:
        """Extract visualization files from a zip archive."""
        logger.info(f"Extracting files from {zip_path.name}")
        
        extract_dir = self.temp_dir / f"extract_{zip_path.stem}"
        os.makedirs(extract_dir, exist_ok=True)
        
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of visualization files
                viz_files = [
                    f for f in zip_ref.namelist()
                    if f.endswith(('.js', '.jsx', '.ts', '.tsx', '.html')) and not '__MACOSX' in f
                ]
                
                # Extract only relevant files
                for i, file in enumerate(viz_files):
                    try:
                        # Skip non-text files
                        try:
                            content = zip_ref.read(file).decode('utf-8')
                        except UnicodeDecodeError:
                            continue
                        
                        extracted_files.append({
                            "file_path": file,
                            "content": content,
                            "source_zip": zip_path.name
                        })
                        
                        # Process in small batches to avoid memory issues
                        if len(extracted_files) >= self.batch_size:
                            batch = extracted_files
                            extracted_files = []
                            yield batch
                        
                        # Report progress
                        if (i + 1) % 100 == 0:
                            logger.info(f"Extracted {i + 1}/{len(viz_files)} files from {zip_path.name}")
                    
                    except Exception as e:
                        logger.error(f"Error extracting {file}: {str(e)}")
                
                # Return any remaining files
                if extracted_files:
                    yield extracted_files
            
        except Exception as e:
            logger.error(f"Error processing zip {zip_path}: {str(e)}")
    
    def process_visualizations(self):
        """Process visualization files."""
        try:
            start_time = time.time()
            
            # Get list of zip files
            zip_files = list(self.input_dir.glob("*.zip"))
            logger.info(f"Found {len(zip_files)} zip files to process")
            
            all_results = []
            
            # Process each zip file
            for zip_path in zip_files:
                if self.state.is_file_processed(str(zip_path)):
                    logger.info(f"Skipping already processed file: {zip_path.name}")
                    continue
                
                logger.info(f"Processing {zip_path.name}")
                self.state.set_current_file(str(zip_path))
                
                zip_results = []
                
                # Process batches of files from the zip
                for batch_idx, batch in enumerate(self._extract_files_from_zip(zip_path)):
                    batch_start = time.time()
                    logger.info(f"Processing batch {batch_idx + 1} from {zip_path.name} ({len(batch)} files)")
                    
                    # Process batch
                    processed_batch = self._process_batch(batch)
                    zip_results.extend(processed_batch)
                    
                    # Save intermediate results every 1000 items
                    if len(zip_results) >= 1000:
                        results_to_save = zip_results[:1000]
                        zip_results = zip_results[1000:]
                        
                        output_file = self.output_dir / f"{zip_path.stem}_batch_{batch_idx}.jsonl"
                        self._save_results(results_to_save, output_file)
                    
                    batch_time = time.time() - batch_start
                    logger.info(f"Batch {batch_idx + 1} processed in {batch_time:.2f}s")
                    
                    # Clean up to reduce memory usage
                    gc.collect()
                
                # Save remaining results for this zip
                if zip_results:
                    output_file = self.output_dir / f"{zip_path.stem}_final.jsonl"
                    self._save_results(zip_results, output_file)
                
                # Record completion
                self.state.mark_file_complete(str(zip_path), len(zip_results))
                all_results.extend(zip_results)
                
                # Clean up extract directory
                extract_dir = self.temp_dir / f"extract_{zip_path.stem}"
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                
                # Force garbage collection
                gc.collect()
            
            # Create final datasets
            if all_results:
                self._create_final_datasets(all_results)
            
            total_time = time.time() - start_time
            logger.info(f"Completed processing in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """Save results to a JSONL file."""
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _create_final_datasets(self, all_results: List[Dict[str, Any]]):
        """Create final datasets for training."""
        logger.info("Creating final datasets")
        
        # Create stats
        stats = {
            "total_items": len(all_results),
            "by_type": {},
            "total_code_size": sum(len(item["code"]) for item in all_results)
        }
        
        # Group by type
        by_type = {}
        for item in all_results:
            viz_type = item["type"]
            
            if viz_type not in by_type:
                by_type[viz_type] = []
                stats["by_type"][viz_type] = 0
            
            by_type[viz_type].append(item)
            stats["by_type"][viz_type] += 1
        
        # Save stats
        with open(self.output_dir / "visualization_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create training corpus
        with open(self.output_dir / "training_corpus.jsonl", 'w') as f:
            for item in all_results:
                training_example = {
                    "instruction": f"Create a {item['type']} visualization",
                    "input": "",
                    "output": item["code"],
                    "metadata": item["metadata"]
                }
                f.write(json.dumps(training_example) + '\n')
        
        logger.info(f"Created final dataset with {len(all_results)} examples")


def process_visualizations(
    input_dir: str = "content/visualisation",
    output_dir: str = "data/visualization",
    max_workers: int = 2,
    batch_size: int = 50
):
    """
    Process visualization files for model training.
    
    Args:
        input_dir: Directory containing visualization zip files
        output_dir: Directory to save processed results
        max_workers: Maximum number of concurrent workers
        batch_size: Number of files to process in each batch
    """
    start_time = time.time()
    logger.info(f"Starting visualization processing from {input_dir} to {output_dir}")
    
    # Create processor
    processor = CodespaceProcessor(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Process visualizations
    processor.process_visualizations()
    
    total_time = time.time() - start_time
    logger.info(f"Visualization processing complete in {total_time:.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process visualization files in Github Codespaces")
    parser.add_argument("--input-dir", type=str, default="content/visualisation",
                       help="Directory containing visualization zip files")
    parser.add_argument("--output-dir", type=str, default="data/visualization",
                       help="Directory to save processed results")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of concurrent workers")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of files to process in each batch")
    
    args = parser.parse_args()
    
    process_visualizations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    ) 