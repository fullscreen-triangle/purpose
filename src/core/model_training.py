"""
Model Training Module.

This module provides functionality for training and fine-tuning language models
on domain-specific data related to sprint running.
"""

import os
import logging
import re
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for processing text data for language model training."""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 512, 
                 stride: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text documents
            tokenizer: Tokenizer for the target model
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize all texts
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            self.examples.extend(self._tokenize_and_chunk(text))
        
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def _tokenize_and_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Tokenize text and split into overlapping chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of tokenized chunks
        """
        tokenized = self.tokenizer(
            text, 
            return_overflowing_tokens=True,
            max_length=self.max_length, 
            stride=self.stride,
            truncation=True, 
            padding="max_length"
        )
        
        examples = []
        for i in range(len(tokenized["input_ids"])):
            examples.append({
                "input_ids": tokenized["input_ids"][i],
                "attention_mask": tokenized["attention_mask"][i]
            })
        
        return examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get an example from the dataset."""
        item = self.examples[idx]
        # Convert lists to tensors
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"])
        }


class ModelTrainer:
    """Trainer for domain-specific language models."""
    
    def __init__(self, 
                 processed_data_dir: str = "data/processed",
                 model_dir: str = "models",
                 model_name: Optional[str] = "distilbert-base-uncased",
                 use_lora: bool = True):
        """
        Initialize the model trainer.
        
        Args:
            processed_data_dir: Directory containing processed text data
            model_dir: Directory to save models
            model_name: Base model name or checkpoint
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_data(self) -> List[str]:
        """
        Load processed text data.
        
        Returns:
            List of text documents
        """
        # Load combined file if it exists
        combined_file = self.processed_data_dir / "all_papers_combined.txt"
        if combined_file.exists():
            logger.info(f"Loading combined text file: {combined_file}")
            with open(combined_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split by document markers
            doc_pattern = r"--- START OF DOCUMENT: (.*?) ---\n\n(.*?)\n\n--- END OF DOCUMENT:"
            docs = []
            for match in re.finditer(doc_pattern, text, re.DOTALL):
                docs.append(match.group(2))
            
            # If we couldn't split properly, treat as one document
            if not docs:
                docs = [text]
            
            logger.info(f"Loaded {len(docs)} documents from combined file")
            return docs
        
        # Otherwise, load individual text files
        logger.info(f"Looking for individual text files in {self.processed_data_dir}")
        text_files = list(self.processed_data_dir.glob("*.txt"))
        
        if not text_files:
            raise FileNotFoundError(f"No text files found in {self.processed_data_dir}")
        
        docs = []
        for file_path in tqdm(text_files, desc="Loading text files"):
            if file_path.name == "all_papers_combined.txt":
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                docs.append(f.read())
        
        logger.info(f"Loaded {len(docs)} individual text files")
        return docs
    
    def prepare_model(self):
        """
        Prepare the model for training.
        
        Returns:
            Prepared model
        """
        logger.info(f"Loading base model: {self.model_name}")
        
        # Load model configuration
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Load base model
        if "gpt" in self.model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                config=config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Setting up LoRA for efficient fine-tuning")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM if "gpt" in self.model_name.lower() else TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,  # rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"] if "gpt" in self.model_name.lower() else ["q_lin", "v_lin"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model
    
    def train(self, 
             batch_size: int = 8, 
             learning_rate: float = 5e-5, 
             num_epochs: int = 3,
             max_seq_length: int = 512,
             save_steps: int = 500):
        """
        Train the language model on domain-specific data.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            max_seq_length: Maximum sequence length
            save_steps: Save checkpoint every N steps
            
        Returns:
            Trained model and training metrics
        """
        # Load data
        texts = self.load_data()
        
        # Prepare dataset
        train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
        
        logger.info("Creating datasets")
        train_dataset = TextDataset(train_texts, self.tokenizer, max_seq_length)
        val_dataset = TextDataset(val_texts, self.tokenizer, max_seq_length)
        
        # Prepare model
        model = self.prepare_model()
        
        # Training arguments
        model_id = f"sprint-llm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.model_dir / model_id
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="steps",
            eval_steps=save_steps // 2,
            save_strategy="steps",
            save_steps=save_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=not "gpt" in self.model_name.lower(),  # MLM for BERT-like, not for GPT-like
            mlm_probability=0.15
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting model training")
        train_result = trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save trainer arguments
        with open(output_dir / "training_args.json", 'w') as f:
            json.dump(training_args.to_dict(), f, indent=2)
        
        logger.info(f"Training complete. Model saved to {output_dir}")
        return model, metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    model, metrics = trainer.train() 