"""
Trainer implementation using Hugging Face Transformers.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import Dataset, load_dataset

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from purpose.trainer.base import ModelTrainer

class TransformersTrainer(ModelTrainer):
    """
    Trainer implementation using Hugging Face Transformers.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "models",
        model_name: str = "gpt2",
        model_output_name: str = "purpose_model",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: Optional[str] = None,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the Transformers trainer.
        
        Args:
            data_dir: Directory containing the processed data
            output_dir: Directory to save the trained model
            model_name: Name or path of the base model to fine-tune
            model_output_name: Name for the output model directory
            use_lora: Whether to use parameter-efficient fine-tuning with LoRA
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout probability
            device: Device to use for training (None for auto-detection)
            seed: Random seed for reproducibility
            **kwargs: Additional trainer-specific arguments
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            use_lora=use_lora,
            device=device,
            **kwargs
        )
        
        self.model_output_name = model_output_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.seed = seed
        
        # Set random seed
        set_seed(self.seed)
        
        # Initialize model and tokenizer to None
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # Check if PEFT is available
        if self.use_lora and not PEFT_AVAILABLE:
            self.logger.warning("LoRA requested but PEFT library not available. "
                                "Installing with: pip install peft")
            self.use_lora = False
    
    def load_data(self, corpus_file: str = "purpose_corpus.txt") -> Dataset:
        """
        Load the training data from the processed corpus.
        
        Args:
            corpus_file: Name of the corpus file in the data directory
            
        Returns:
            Hugging Face Dataset with the loaded data
        """
        corpus_path = self.data_dir / corpus_file
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        self.logger.info(f"Loading corpus from {corpus_path}")
        
        # Read the corpus file
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create a simple dataset
        self.dataset = Dataset.from_dict({"text": [text]})
        
        return self.dataset
    
    def _tokenize_dataset(self, max_length: int = 512):
        """
        Tokenize the dataset for training.
        
        Args:
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        self.logger.info("Tokenizing dataset")
        
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
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        return tokenized_dataset
    
    def _initialize_model(self):
        """
        Initialize the model for training.
        """
        self.logger.info(f"Initializing model: {self.model_name}")
        
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize the base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Apply LoRA if requested
        if self.use_lora and PEFT_AVAILABLE:
            self.logger.info("Applying LoRA for parameter-efficient fine-tuning")
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "v_proj"]  # Target attention layers
            )
            
            # Get PEFT model
            self.model = get_peft_model(self.model, peft_config)
            
        # Move model to device
        self.model.to(self.device)
    
    def train(
        self,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        max_length: int = 512,
        save_steps: int = 1000,
        logging_steps: int = 100,
        **kwargs
    ) -> None:
        """
        Train the model on the processed data.
        
        Args:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate scheduler
            max_length: Maximum sequence length
            save_steps: Save checkpoint every N steps
            logging_steps: Log training metrics every N steps
            **kwargs: Additional training arguments
        """
        try:
            # Load dataset if not already loaded
            if self.dataset is None:
                self.load_data()
            
            # Tokenize dataset
            tokenized_dataset = self._tokenize_dataset(max_length=max_length)
            
            # Initialize model
            self._initialize_model()
            
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
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=2,
                prediction_loss_only=True,
                fp16=torch.cuda.is_available(),  # Use mixed precision if available
                **kwargs
            )
            
            # Set up data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling, not masked
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )
            
            # Train the model
            self.logger.info("Starting training")
            self.trainer.train()
            
            # Save the final model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            output_path: Path to save the model (default: output_dir/model_output_name)
            
        Returns:
            Path where the model was saved
        """
        # Default output path
        if output_path is None:
            output_path = str(self.output_dir / self.model_output_name)
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save model and tokenizer
        if self.trainer is not None and self.model is not None and self.tokenizer is not None:
            self.trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)
            self.logger.info(f"Model saved to {output_path}")
        else:
            self.logger.warning("Cannot save model: trainer, model, or tokenizer is None")
        
        return output_path
    
    def evaluate(self, test_data=None, **kwargs):
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data: Test data for evaluation
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is not None:
            if test_data is not None:
                # Tokenize test data if provided
                if isinstance(test_data, str):
                    # Test data is a corpus file
                    test_corpus_path = Path(test_data)
                    with open(test_corpus_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    test_dataset = Dataset.from_dict({"text": [text]})
                else:
                    # Test data is already a dataset
                    test_dataset = test_data
                
                # Tokenize test dataset
                tokenize_function = lambda examples: self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_special_tokens_mask=True,
                )
                
                tokenized_test_dataset = test_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"],
                )
                
                # Evaluate
                metrics = self.trainer.evaluate(eval_dataset=tokenized_test_dataset)
                self.logger.info(f"Evaluation results: {metrics}")
                return metrics
            else:
                self.logger.warning("No test data provided for evaluation")
        else:
            self.logger.warning("Cannot evaluate: trainer is None")
        
        return None 