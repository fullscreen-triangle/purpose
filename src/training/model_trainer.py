"""
Model Trainer for Sprint Results LLM

This module provides functionality to fine-tune a language model on sprint results data.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model-trainer")

class SprintModelTrainer:
    """Trainer for fine-tuning language models on sprint results data."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dir: str = "models",
        data_dir: str = "data/processed",
        use_lora: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name of the base model to fine-tune
            output_dir: Directory to save the trained model
            data_dir: Directory containing the processed data
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.use_lora = use_lora
        self.seed = seed
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed
        set_seed(self.seed)
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def load_corpus(self, corpus_file: str = "sprint_corpus.txt") -> Dataset:
        """
        Load and prepare the text corpus for training.
        
        Args:
            corpus_file: Name of the corpus file in the data directory
            
        Returns:
            Dataset object ready for training
        """
        corpus_path = self.data_dir / corpus_file
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        logger.info(f"Loading corpus from {corpus_path}")
        
        # Read the corpus file
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create a simple dataset
        dataset = Dataset.from_dict({"text": [text]})
        
        return dataset
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize the dataset for training.
        
        Args:
            dataset: Raw dataset with text
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Tokenizing dataset")
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True,
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        return tokenized_dataset
    
    def initialize_model(self):
        """Initialize the model for training."""
        logger.info(f"Initializing model: {self.model_name}")
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize the base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA for parameter-efficient fine-tuning")
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,  # Rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]  # Target attention layers
            )
            
            # Get PEFT model
            self.model = get_peft_model(self.model, peft_config)
            
        # Move model to device
        self.model.to(self.device)
    
    def train(
        self,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        save_steps: int = 1000,
    ) -> None:
        """
        Train the model on the sprint results corpus.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
        """
        # Load and prepare dataset
        try:
            dataset = self.load_corpus()
            tokenized_dataset = self.prepare_dataset(dataset)
            
            # Initialize model
            self.initialize_model()
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                weight_decay=0.01,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=100,
                save_steps=save_steps,
                save_total_limit=2,
                prediction_loss_only=True,
                fp16=torch.cuda.is_available(),  # Use mixed precision if available
            )
            
            # Set up data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling, not masked
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )
            
            # Train the model
            logger.info("Starting training")
            trainer.train()
            
            # Save the final model
            model_save_path = self.output_dir / "sprint_model"
            trainer.save_model(str(model_save_path))
            self.tokenizer.save_pretrained(str(model_save_path))
            
            logger.info(f"Model saved to {model_save_path}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def train_model(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    epochs: int = 3,
    use_lora: bool = False,
) -> None:
    """
    Train a language model on sprint results data.
    
    Args:
        data_dir: Directory containing the processed data
        model_dir: Directory to save the trained model
        model_name: Name of the base model to fine-tune
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Number of training epochs
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
    """
    try:
        # Initialize trainer
        trainer = SprintModelTrainer(
            model_name=model_name,
            output_dir=model_dir,
            data_dir=data_dir,
            use_lora=use_lora,
        )
        
        # Train the model
        trainer.train(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise 