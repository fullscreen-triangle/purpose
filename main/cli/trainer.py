"""
Enhanced Distributed Trainer

This module contains the core training functionality using Ray and Dask
for distributed processing.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import gc
import psutil
from functools import partial

import torch
import ray
# Fix Ray imports - the API has changed
# from ray import train
# from ray.train import Trainer
# from ray.train.torch import TorchTrainer
import dask.dataframe as dd
import dask.bag as db
from dask.distributed import Client, LocalCluster
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manage system resources for distributed training."""
    
    def __init__(self, memory_fraction: float = 0.8):
        """Initialize resource manager."""
        self.memory_fraction = memory_fraction
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = self.total_memory * self.memory_fraction
        self.num_cpus = psutil.cpu_count(logical=False)
        self.gpu_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size for Dask based on available memory."""
        # Aim for chunks that use 5% of available memory
        chunk_memory = self.available_memory * 0.05
        return int(chunk_memory / 1024)  # Convert to KB
    
    def get_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        # Reserve 1 CPU for system tasks
        return max(1, self.num_cpus - 1)
    
    def get_ray_resources(self) -> Dict[str, Any]:
        """Get resource configuration for Ray."""
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory": int(self.available_memory * 0.8)  # 80% of available memory
        }
    
    def setup_dask_client(self) -> Client:
        """Setup Dask client with optimal configuration."""
        cluster = LocalCluster(
            n_workers=self.get_optimal_workers(),
            threads_per_worker=2,
            memory_limit=f"{int(self.available_memory / self.get_optimal_workers())}B"
        )
        return Client(cluster)

class DistributedDataProcessor:
    """Process and prepare data for distributed training."""
    
    def __init__(
        self,
        data_dir: Path,
        resource_manager: ResourceManager,
        max_examples_per_domain: int = 100000
    ):
        """Initialize data processor."""
        self.data_dir = data_dir
        self.resource_manager = resource_manager
        self.max_examples = max_examples_per_domain
        self.dask_client = resource_manager.setup_dask_client()
    
    def process_text_file(self, file_path: Path) -> db.Bag:
        """Process text file using Dask."""
        chunk_size = self.resource_manager.get_optimal_chunk_size()
        return db.read_text(str(file_path)).map_partitions(
            lambda x: [{"text": line} for line in x]
        ).take(self.max_examples)
    
    def process_json_file(self, file_path: Path) -> db.Bag:
        """Process JSON file using Dask."""
        return db.read_text(str(file_path)).map(json.loads).take(self.max_examples)
    
    def load_and_process_data(self) -> Dataset:
        """Load and process all data sources using Dask."""
        datasets = []
        
        # Process purpose corpus
        purpose_path = self.data_dir / "processed" / "purpose_corpus.txt"
        if purpose_path.exists():
            try:
                purpose_data = self.process_text_file(purpose_path)
                datasets.extend(purpose_data)
                logger.info(f"Processed purpose corpus: {len(purpose_data)} examples")
            except Exception as e:
                logger.error(f"Error processing purpose corpus: {str(e)}")
        
        # Process enhanced data
        enhanced_path = self.data_dir / "enhanced" / "enhanced_models.json"
        if enhanced_path.exists():
            try:
                enhanced_data = self.process_json_file(enhanced_path)
                datasets.extend([
                    {"text": f"DOMAIN: {item['domain']}\nMODEL:\n{item['model_result']}\n"}
                    for item in enhanced_data
                ])
                logger.info(f"Processed enhanced data: {len(enhanced_data)} examples")
            except Exception as e:
                logger.error(f"Error processing enhanced data: {str(e)}")
        
        # Process visualization data
        viz_path = self.data_dir / "visualization" / "training_corpus.jsonl"
        if viz_path.exists():
            try:
                viz_data = self.process_json_file(viz_path)
                datasets.extend([
                    {"text": f"TASK: {item['instruction']}\n\nCODE:\n{item['output']}\n"}
                    for item in viz_data
                ])
                logger.info(f"Processed visualization data: {len(viz_data)} examples")
            except Exception as e:
                logger.error(f"Error processing visualization data: {str(e)}")
        
        if not datasets:
            raise ValueError("No training data found!")
        
        # Convert to HuggingFace Dataset
        return Dataset.from_list(datasets)

@ray.remote
class DistributedTrainer:
    """Distributed model training using Ray."""
    
    def __init__(
        self,
        model_name: str,
        tokenizer,
        lora_config: Optional[LoraConfig] = None,
        max_length: int = 1024
    ):
        """Initialize trainer on a Ray worker."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if lora_config:
            self.model = get_peft_model(self.model, lora_config)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def train_batch(self, batch_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Train on a batch of data."""
        inputs = self.tokenizer(
            [item["text"] for item in batch_data],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.model = self.model.cuda()
        
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        loss.backward()
        
        return {"loss": loss.item()}

class EnhancedDistributedTrainer:
    """Main class for distributed training coordination."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        output_name: str = "enhanced_model",
        data_dir: str = "data",
        model_dir: str = "models",
        use_lora: bool = True,
        memory_fraction: float = 0.8
    ):
        """Initialize distributed training coordinator."""
        self.model_name = model_name
        self.output_name = output_name
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.use_lora = use_lora
        
        # Initialize resource management
        self.resource_manager = ResourceManager(memory_fraction)
        
        # Initialize Ray
        ray.init(**self.resource_manager.get_ray_resources())
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup LoRA if enabled
        self.lora_config = None
        if use_lora:
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """Run distributed training."""
        try:
            # Initialize data processor
            data_processor = DistributedDataProcessor(
                self.data_dir,
                self.resource_manager
            )
            
            # Load and process data
            dataset = data_processor.load_and_process_data()
            logger.info(f"Loaded {len(dataset)} examples for training")
            
            # Create distributed trainers
            num_workers = self.resource_manager.get_optimal_workers()
            trainers = [
                DistributedTrainer.remote(
                    self.model_name,
                    self.tokenizer,
                    self.lora_config
                )
                for _ in range(num_workers)
            ]
            
            # Training loop
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Shuffle dataset
                dataset = dataset.shuffle()
                
                # Create batches
                batches = [
                    dataset[i:i + batch_size]
                    for i in range(0, len(dataset), batch_size)
                ]
                
                # Distribute batches across workers
                futures = []
                for i, batch in enumerate(batches):
                    trainer_idx = i % num_workers
                    futures.append(trainers[trainer_idx].train_batch.remote(batch))
                
                # Collect results
                results = ray.get(futures)
                epoch_loss = np.mean([r["loss"] for r in results])
                logger.info(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")
                
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save model
            output_path = self.model_dir / self.output_name
            os.makedirs(output_path, exist_ok=True)
            
            # Get model from first trainer and save
            trainer = ray.get(trainers[0])
            trainer.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        
        finally:
            # Cleanup
            ray.shutdown()
            data_processor.dask_client.close()