"""
Distributed Data Processor

This module implements distributed processing for large visualization datasets.
It uses chunked processing, disk-based operations, and checkpointing to handle
large files efficiently without overwhelming system memory.
"""

import os
import json
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import time
from datetime import datetime
import hashlib
import numpy as np

import dask.bag as db
from dask.distributed import Client, LocalCluster
import ray
import psutil
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distributed_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingState:
    """Manages processing state and checkpointing."""
    
    def __init__(self, state_dir: Path):
        """Initialize processing state."""
        self.state_dir = state_dir
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
            "current_position": 0,
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
        self.state["current_position"] = 0
        self.state["total_processed"] += processed_items
        self.save_state()
    
    def is_file_processed(self, file_path: str) -> bool:
        """Check if a file has been processed."""
        return file_path in self.state["processed_files"]
    
    def set_current_file(self, file_path: str, position: int = 0):
        """Set the current file being processed."""
        self.state["current_file"] = file_path
        self.state["current_position"] = position
        self.save_state()

class ResourceManager:
    """Manages system resources for distributed processing."""
    
    def __init__(self, memory_fraction: float = 0.4):
        """Initialize resource manager."""
        self.memory_fraction = memory_fraction
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = self.total_memory * self.memory_fraction
        self.num_cpus = psutil.cpu_count(logical=False)
        
        # More conservative memory settings
        self.chunk_size = int(self.available_memory * 0.05)  # 5% of available memory per chunk
        self.max_workers = max(1, min(self.num_cpus - 2, 4))  # More conservative worker count
    
    def get_dask_client(self) -> Client:
        """Create a Dask client with optimal settings."""
        memory_per_worker = int(self.available_memory / self.max_workers)
        
        cluster = LocalCluster(
            n_workers=self.max_workers,
            threads_per_worker=2,
            memory_limit=f"{memory_per_worker}B",
            memory_target_fraction=0.7  # Target 70% of allocated memory
        )
        return Client(cluster)
    
    def get_ray_config(self) -> Dict[str, Any]:
        """Get Ray configuration."""
        return {
            "num_cpus": self.max_workers,
            "memory": int(self.available_memory * 0.7),  # 70% of allocated memory for Ray
            "_temp_dir": str(Path.home() / ".purpose" / "ray_temp"),
            "object_store_memory": int(self.available_memory * 0.2)  # Limit object store
        }

@ray.remote
class ChunkProcessor:
    """Processes chunks of data using Ray."""
    
    def process_chunk(self, chunk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of data."""
        processed = []
        
        for item in chunk_data:
            try:
                # Extract relevant information
                processed_item = {
                    "type": self._detect_visualization_type(item),
                    "code": self._extract_code(item),
                    "metadata": self._extract_metadata(item)
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
        
        return processed
    
    def _detect_visualization_type(self, item: Dict[str, Any]) -> str:
        """Detect the type of visualization from the code."""
        code = item.get("code", "").lower()
        
        if "d3" in code:
            if "force" in code:
                return "force_directed"
            elif "tree" in code:
                return "tree"
            elif "map" in code:
                return "map"
            return "d3_other"
            
        if "react" in code:
            return "react"
            
        return "other"
    
    def _extract_code(self, item: Dict[str, Any]) -> str:
        """Extract and clean visualization code."""
        code = item.get("code", "")
        
        # Remove comments
        code_lines = []
        for line in code.split("\n"):
            line = line.strip()
            if line and not line.startswith("//") and not line.startswith("/*"):
                code_lines.append(line)
        
        return "\n".join(code_lines)
    
    def _extract_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the item."""
        return {
            "file_type": item.get("file_type"),
            "dependencies": item.get("dependencies", []),
            "size": len(item.get("code", "")),
            "has_styles": "style" in item.get("code", "").lower()
        }

class DistributedProcessor:
    """Main class for distributed processing of visualization data."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        temp_dir: Optional[Path] = None,
        memory_fraction: float = 0.6,
        chunk_size_mb: int = 100
    ):
        """Initialize distributed processor."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="purpose_processing_"))
        self.chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize components
        self.resource_manager = ResourceManager(memory_fraction)
        self.state = ProcessingState(self.temp_dir)
        
        # Initialize Ray
        ray.init(**self.resource_manager.get_ray_config())
    
    def _extract_zip_chunked(self, zip_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Extract and process zip files in chunks."""
        extract_dir = self.temp_dir / f"extract_{zip_path.stem}"
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of visualization files
            viz_files = [
                f for f in zip_ref.namelist()
                if f.endswith(('.js', '.jsx', '.ts', '.tsx', '.html'))
            ]
            
            current_chunk = []
            current_size = 0
            
            for file in viz_files:
                try:
                    # Extract file
                    zip_ref.extract(file, extract_dir)
                    file_path = extract_dir / file
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create item
                    item = {
                        "file_path": file,
                        "code": content,
                        "file_type": file_path.suffix,
                        "source_zip": zip_path.name
                    }
                    
                    current_chunk.append(item)
                    current_size += len(content)
                    
                    # If chunk is full, yield it
                    if current_size >= self.chunk_size:
                        yield current_chunk
                        current_chunk = []
                        current_size = 0
                    
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
            
            # Yield remaining items
            if current_chunk:
                yield current_chunk
    
    def process_visualization_data(self):
        """Process visualization data using distributed computing."""
        try:
            # Get list of zip files
            zip_files = list(self.input_dir.glob("*.zip"))
            logger.info(f"Found {len(zip_files)} zip files to process")
            
            # Create chunk processors
            processors = [
                ChunkProcessor.remote()
                for _ in range(self.resource_manager.max_workers)
            ]
            
            # Process each zip file
            for zip_path in zip_files:
                if self.state.is_file_processed(str(zip_path)):
                    logger.info(f"Skipping already processed file: {zip_path.name}")
                    continue
                
                logger.info(f"Processing {zip_path.name}")
                self.state.set_current_file(str(zip_path))
                
                # Process chunks
                chunk_results = []
                for chunk_idx, chunk in enumerate(self._extract_zip_chunked(zip_path)):
                    # Distribute chunks to processors
                    futures = []
                    for i, item_batch in enumerate(
                        np.array_split(chunk, len(processors))
                    ):
                        if len(item_batch) > 0:
                            futures.append(
                                processors[i].process_chunk.remote(item_batch.tolist())
                            )
                    
                    # Collect results
                    results = ray.get(futures)
                    chunk_results.extend([
                        item for sublist in results
                        for item in sublist
                    ])
                    
                    # Save intermediate results
                    if len(chunk_results) >= 1000:
                        self._save_results(
                            chunk_results,
                            f"{zip_path.stem}_chunk_{chunk_idx}.jsonl"
                        )
                        chunk_results = []
                
                # Save remaining results
                if chunk_results:
                    self._save_results(
                        chunk_results,
                        f"{zip_path.stem}_final.jsonl"
                    )
                
                # Mark file as processed
                self.state.mark_file_complete(str(zip_path), len(chunk_results))
                
                # Clear temporary files
                shutil.rmtree(self.temp_dir / f"extract_{zip_path.stem}")
            
            # Combine all results
            self._combine_results()
            
            logger.info("Processing complete!")
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise
            
        finally:
            # Cleanup
            ray.shutdown()
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save processing results to disk."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
    
    def _combine_results(self):
        """Combine all processed results into final datasets."""
        # Combine all JSONL files
        all_results = []
        for jsonl_file in self.output_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
        
        # Create final datasets
        final_data = {
            "all": all_results,
            "by_type": {},
            "stats": {
                "total_items": len(all_results),
                "by_type": {},
                "total_code_size": sum(len(item["code"]) for item in all_results)
            }
        }
        
        # Group by visualization type
        for item in all_results:
            viz_type = item["type"]
            if viz_type not in final_data["by_type"]:
                final_data["by_type"][viz_type] = []
            final_data["by_type"][viz_type].append(item)
            
            # Update stats
            if viz_type not in final_data["stats"]["by_type"]:
                final_data["stats"]["by_type"][viz_type] = 0
            final_data["stats"]["by_type"][viz_type] += 1
        
        # Save final datasets
        with open(self.output_dir / "visualization_dataset.json", 'w') as f:
            json.dump(final_data, f, indent=2)
        
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