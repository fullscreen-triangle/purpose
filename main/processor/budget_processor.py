#!/usr/bin/env python3
"""
Budget-Friendly Visualization Processor

This script processes visualization files one at a time with minimal memory usage.
Designed to run on the smallest possible Codespace configuration to minimize costs.
"""

import os
import json
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
import time
import re
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("budget_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BudgetProcessor:
    """
    Processes visualization data one file at a time with minimal resource usage.
    """
    
    def __init__(
        self,
        input_dir: str = "content/visualisation",
        output_dir: str = "data/visualization",
        temp_dir: str = None,
        max_files_per_zip: int = 100,  # Process only a subset of files from each zip
        file_chunk_size: int = 5000000,  # ~5MB chunks
        max_spending_euros: float = 20.0  # Hard spending limit
    ):
        """Initialize budget processor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="budget_processing_"))
        self.max_files_per_zip = max_files_per_zip
        self.file_chunk_size = file_chunk_size
        self.max_spending_euros = max_spending_euros
        self.state_file = self.temp_dir / "processing_state.json"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load state
        self.state = self._load_state()
    
    def _load_state(self):
        """Load processing state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        return {
            "processed_zips": [],
            "current_zip": None,
            "processed_count": 0,
            "last_update": None
        }
    
    def _save_state(self):
        """Save current processing state to disk."""
        self.state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def _detect_visualization_type(self, code):
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
            elif "scatter" in code_lower:
                return "scatter_plot"
            return "d3_other"
            
        if "react" in code_lower:
            return "react"
        
        return "other"
    
    def _extract_metadata(self, file_path, code):
        """Extract basic metadata from the file."""
        file_type = Path(file_path).suffix
        
        # Simple library detection via imports
        libraries = []
        if "import" in code and "from" in code:
            libraries.append("es6")
        if "require(" in code:
            libraries.append("commonjs")
        if "d3" in code.lower():
            libraries.append("d3")
        if "react" in code.lower():
            libraries.append("react")
        
        return {
            "file_type": file_type,
            "size": len(code),
            "libraries": libraries,
            "has_styles": "style" in code.lower() or "css" in code.lower()
        }
    
    def _clean_code(self, code):
        """Minimally clean the code to save memory."""
        # Remove comments - simple version to save CPU
        code = re.sub(r'\/\/.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)
        
        return code.strip()
    
    def _process_file(self, file_path, content):
        """Process a single file with minimal memory usage."""
        try:
            # Clean the code
            cleaned_code = self._clean_code(content)
            
            # Basic detection
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
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _process_zip_file(self, zip_path):
        """Process a single zip file."""
        logger.info(f"Processing {zip_path.name}")
        self.state["current_zip"] = str(zip_path)
        self._save_state()
        
        results = []
        processed_count = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get visualization files
                viz_files = [
                    f for f in zip_ref.namelist()
                    if f.endswith(('.js', '.jsx', '.ts', '.tsx', '.html')) and 
                    not '__MACOSX' in f
                ]
                
                logger.info(f"Found {len(viz_files)} visualization files in {zip_path.name}")
                
                # Process limited number of files
                files_to_process = viz_files[:self.max_files_per_zip]
                logger.info(f"Will process up to {len(files_to_process)} files from {zip_path.name}")
                
                for i, file in enumerate(files_to_process):
                    try:
                        # Only check progress every 10 files to save time
                        if i % 10 == 0:
                            logger.info(f"Progress: {i}/{len(files_to_process)} files from {zip_path.name}")
                        
                        # Try to decode the file
                        try:
                            content = zip_ref.read(file).decode('utf-8')
                        except UnicodeDecodeError:
                            continue
                        
                        # Skip very large files that could cause memory issues
                        if len(content) > self.file_chunk_size:
                            logger.warning(f"Skipping large file {file} ({len(content)} bytes)")
                            continue
                        
                        # Process the file
                        result = self._process_file(file, content)
                        if result:
                            results.append(result)
                            processed_count += 1
                        
                        # Save intermediate results every 25 files to avoid memory buildup
                        if len(results) >= 25:
                            self._save_results(results, f"{zip_path.stem}_part_{i//25}.jsonl")
                            results = []
                            # Force garbage collection
                            gc.collect()
                    
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {e}")
            
            # Save any remaining results
            if results:
                self._save_results(results, f"{zip_path.stem}_final.jsonl")
            
            # Update state
            self.state["processed_zips"].append(str(zip_path))
            self.state["current_zip"] = None
            self.state["processed_count"] += processed_count
            self._save_state()
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing zip {zip_path}: {e}")
            return 0
    
    def _save_results(self, results, filename):
        """Save results to a JSONL file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def process(self, max_zips=None, time_limit_minutes=60):
        """
        Process visualization files with budget constraints.
        
        Args:
            max_zips: Maximum number of zip files to process
            time_limit_minutes: Stop processing after this many minutes
        """
        start_time = time.time()
        end_time = start_time + (time_limit_minutes * 60)
        
        # Calculate maximum runtime based on spending limit
        # €0.18 per hour for 2-core Codespace
        max_hours_from_budget = self.max_spending_euros / 0.18
        max_seconds_from_budget = max_hours_from_budget * 3600
        budget_end_time = start_time + max_seconds_from_budget
        
        # Use the stricter of the two limits
        end_time = min(end_time, budget_end_time)
        
        logger.info(f"Will stop processing after {time_limit_minutes} minutes or when spending approaches €{self.max_spending_euros}")
        logger.info(f"Maximum processing time allowed: {max_hours_from_budget:.2f} hours to stay under €{self.max_spending_euros}")
        
        try:
            # Get list of zip files
            zip_files = list(self.input_dir.glob("*.zip"))
            logger.info(f"Found {len(zip_files)} zip files")
            
            # Skip previously processed zip files
            zip_files = [z for z in zip_files if str(z) not in self.state["processed_zips"]]
            logger.info(f"{len(zip_files)} zip files remain to be processed")
            
            # Apply max_zips constraint if specified
            if max_zips is not None and max_zips > 0:
                zip_files = zip_files[:max_zips]
                logger.info(f"Will process up to {max_zips} zip files due to budget constraints")
            
            # Process each zip file
            for i, zip_path in enumerate(zip_files):
                # Check time limit
                current_time = time.time()
                if current_time > end_time:
                    elapsed_hours = (current_time - start_time) / 3600
                    estimated_cost = elapsed_hours * 0.18
                    remaining_budget = self.max_spending_euros - estimated_cost
                    
                    if remaining_budget <= 0:
                        logger.warning(f"STOPPING: Budget limit of €{self.max_spending_euros} reached!")
                    else:
                        logger.info(f"Stopping due to time limit. Estimated cost so far: €{estimated_cost:.2f}")
                    
                    break
                
                # Calculate and log cost estimate
                elapsed_minutes = (current_time - start_time) / 60
                elapsed_hours = elapsed_minutes / 60
                approx_cost = elapsed_hours * 0.18  # ~€0.18 per hour for 2-core Codespace
                
                # Check if we're approaching budget limit
                if approx_cost >= (self.max_spending_euros * 0.9):  # 90% of limit
                    logger.warning(f"APPROACHING BUDGET LIMIT! Current cost: €{approx_cost:.2f}, limit: €{self.max_spending_euros}")
                    
                logger.info(f"Processed {i} zip files. Elapsed time: {elapsed_minutes:.1f} min. Cost: €{approx_cost:.2f} of €{self.max_spending_euros} limit")
                
                # Process zip file
                processed = self._process_zip_file(zip_path)
                logger.info(f"Completed {zip_path.name}: processed {processed} files")
                
                # Force garbage collection
                gc.collect()
                
                # Check budget after each zip file
                current_time = time.time()
                elapsed_hours = (current_time - start_time) / 3600
                current_cost = elapsed_hours * 0.18
                
                if current_cost >= (self.max_spending_euros * 0.95):  # 95% of limit as safety margin
                    logger.warning(f"STOPPING: Approaching budget limit of €{self.max_spending_euros}. Current cost: €{current_cost:.2f}")
                    break
            
            # Create final dataset if time allows and we haven't reached budget limit
            current_time = time.time()
            elapsed_hours = (current_time - start_time) / 3600
            current_cost = elapsed_hours * 0.18
            
            if current_time < end_time and current_cost < (self.max_spending_euros * 0.95):
                self._create_final_dataset()
            
            # Log final cost estimate
            total_minutes = (time.time() - start_time) / 60
            total_hours = total_minutes / 60
            total_cost = total_hours * 0.18
            
            logger.info(f"Total processing time: {total_minutes:.1f} minutes ({total_hours:.2f} hours)")
            logger.info(f"Estimated cost: €{total_cost:.2f} of €{self.max_spending_euros} limit")
            
            if total_cost > (self.max_spending_euros * 0.9):
                logger.warning(f"You've used {(total_cost/self.max_spending_euros)*100:.1f}% of your budget limit!")
            
        except Exception as e:
            # Log the error and the current cost estimate
            elapsed_hours = (time.time() - start_time) / 3600
            current_cost = elapsed_hours * 0.18
            logger.error(f"Processing failed: {e}")
            logger.info(f"Estimated cost before error: €{current_cost:.2f}")
            raise
    
    def _create_final_dataset(self):
        """Create a final dataset from processed files."""
        logger.info("Creating final dataset")
        
        # Find all JSONL files
        jsonl_files = list(self.output_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.warning("No JSONL files found to create final dataset")
            return
        
        # We'll just create a simple instruction dataset to save memory
        with open(self.output_dir / "training_corpus.jsonl", 'w') as outfile:
            item_count = 0
            
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r') as infile:
                        for line in infile:
                            # Parse each line
                            item = json.loads(line)
                            
                            # Create a training example
                            example = {
                                "instruction": f"Create a {item['type']} visualization",
                                "input": "",
                                "output": item["code"]
                            }
                            
                            # Write to output file
                            outfile.write(json.dumps(example) + '\n')
                            item_count += 1
                            
                            # Log progress occasionally
                            if item_count % 100 == 0:
                                logger.info(f"Processed {item_count} items for training corpus")
                except Exception as e:
                    logger.error(f"Error processing file {jsonl_file}: {e}")
        
        logger.info(f"Created training corpus with {item_count} examples")


def main():
    """Run the budget processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Budget-friendly visualization processing")
    parser.add_argument("--input-dir", default="content/visualisation", help="Directory with zip files")
    parser.add_argument("--output-dir", default="data/visualization", help="Output directory")
    parser.add_argument("--max-zips", type=int, default=1, help="Maximum number of zip files to process")
    parser.add_argument("--max-files", type=int, default=100, help="Maximum files per zip")
    parser.add_argument("--time-limit", type=int, default=60, help="Time limit in minutes")
    parser.add_argument("--max-spending", type=float, default=20.0, help="Maximum spending limit in euros (default: 20.0)")
    
    args = parser.parse_args()
    
    processor = BudgetProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files_per_zip=args.max_files,
        max_spending_euros=args.max_spending
    )
    
    processor.process(max_zips=args.max_zips, time_limit_minutes=args.time_limit)


if __name__ == "__main__":
    main() 