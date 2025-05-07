#!/usr/bin/env python3
"""
Enhanced Training Module

This module implements the enhanced training functionality for domain-specific models
using knowledge distillation and curriculum learning as outlined in knowledge.md.
It provides a comprehensive pipeline for creating specialized models from research papers.
"""

import os
import logging
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset, load_dataset

from main.trainer.enhanced_distillation import (
    extract_knowledge,
    create_knowledge_map,
    generate_qa_pairs,
    generate_enhanced_responses,
    create_curriculum_dataset,
    run_enhanced_distillation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced-training")

class EnhancedTrainer:
    """
    Enhanced trainer class that implements knowledge distillation and model optimization
    for creating domain-specific models using large language models as teachers.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        domain_config: str,
        target_model: str = "distilgpt2",
        use_lora: bool = True,
        lora_r: int = 8,
        device: str = "auto"
    ):
        """
        Initialize the Enhanced Trainer.
        
        Args:
            data_dir: Directory containing domain data (papers, etc.)
            output_dir: Directory to save processed data and models
            domain_config: Path to domain configuration file
            target_model: Base model to fine-tune
            use_lora: Whether to use LoRA for fine-tuning
            lora_r: LoRA attention dimension
            device: Device to use for training ('cpu', 'cuda', or 'auto')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.domain_config_path = domain_config
        self.target_model = target_model
        self.use_lora = use_lora
        self.lora_r = lora_r
        
        # Create necessary directories
        self.papers_dir = self.data_dir / "papers"
        self.processed_data_dir = self.output_dir / "processed_data"
        self.models_dir = self.output_dir / "models"
        
        for dir_path in [self.processed_data_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Load domain configuration
        self.domain_config = self._load_domain_config()
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Enhanced Trainer initialized for domain: {self.domain_config.get('domain_name', 'unknown')}")
        logger.info(f"Using device: {self.device}")
        
    def _load_domain_config(self) -> Dict:
        """
        Load domain configuration from the specified file.
        
        Returns:
            Dictionary containing domain configuration
        """
        try:
            with open(self.domain_config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded domain configuration from {self.domain_config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading domain configuration: {e}")
            logger.warning("Using default configuration")
            return {
                "domain_name": "generic",
                "description": "Generic domain",
                "key_concepts": [],
                "target_audience": "general",
                "training_parameters": {
                    "batch_size": 4,
                    "learning_rate": 5e-5,
                    "num_epochs": 3
                }
            }
            
    def generate_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Generate the base model and tokenizer for training.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading base model: {self.target_model}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.target_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Determine if we need quantization
            quantize = False
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
                model_size_map = {
                    "distilgpt2": 0.5,
                    "gpt2": 0.5,
                    "gpt2-medium": 1.5,
                    "gpt2-large": 3.0,
                    "gpt2-xl": 6.0,
                    "EleutherAI/pythia-1b": 2.0,
                    "EleutherAI/pythia-6.9b": 14.0,
                    "meta-llama/Llama-2-7b": 14.0
                }
                
                model_size = model_size_map.get(self.target_model, 2.0)  # Default assumption
                if gpu_memory < model_size * 2:  # Need ~2x model size for training
                    quantize = True
                    logger.info(f"GPU memory ({gpu_memory:.2f} GB) too small for full precision model. Using quantization.")
            
            # Load model with appropriate settings
            if quantize:
                if "llama" in self.target_model.lower():
                    from transformers import BitsAndBytesConfig
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        self.target_model,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.target_model,
                        load_in_8bit=True,
                        device_map="auto"
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(self.target_model)
                if self.device != "cpu":
                    model = model.to(self.device)
            
            # Apply LoRA if requested
            if self.use_lora:
                # Determine target modules based on model architecture
                if "llama" in self.target_model.lower() or "mistral" in self.target_model.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif "gpt2" in self.target_model.lower() or "pythia" in self.target_model.lower():
                    target_modules = ["c_attn", "c_proj", "c_fc"]
                else:
                    # Default modules that work for many transformer architectures
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
                lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=32,
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                
                logger.info(f"Applying LoRA configuration with r={self.lora_r}")
                model = get_peft_model(model, lora_config)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error generating model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    def create_training_corpus(self, curriculum_path: str) -> Dict[str, Dataset]:
        """
        Create a training corpus from the curriculum dataset.
        
        Args:
            curriculum_path: Path to curriculum dataset
            
        Returns:
            Dictionary mapping stage names to datasets
        """
        logger.info(f"Creating training corpus from {curriculum_path}")
        
        try:
            with open(curriculum_path, 'r') as f:
                curriculum = json.load(f)
                
            tokenizer = AutoTokenizer.from_pretrained(self.target_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Process each curriculum stage
            datasets = {}
            
            for stage_name, stage_data in curriculum["stages"].items():
                logger.info(f"Processing {stage_name} stage with {len(stage_data)} examples")
                
                # Create formatted examples
                formatted_examples = []
                
                for qa_pair in stage_data:
                    question = qa_pair["question"]
                    answer = qa_pair["answer"]
                    
                    # Format as instruction fine-tuning
                    formatted_text = f"Question: {question}\n\nAnswer: {answer}"
                    formatted_examples.append({"text": formatted_text})
                
                # Create dataset
                stage_dataset = Dataset.from_list(formatted_examples)
                
                # Tokenize dataset
                def tokenize_function(examples):
                    return tokenizer(
                        examples["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=1024,
                        return_tensors="pt"
                    )
                
                tokenized_dataset = stage_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"]
                )
                
                datasets[stage_name] = tokenized_dataset
                
            return datasets
            
        except Exception as e:
            logger.error(f"Error creating training corpus: {e}")
            return {}
            
    def train_student_model(
        self,
        datasets: Dict[str, Dataset],
        model,
        tokenizer,
        output_dir: str,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3
    ) -> str:
        """
        Train the student model using the curriculum datasets.
        
        Args:
            datasets: Dictionary of datasets for each curriculum stage
            model: Model to train
            tokenizer: Tokenizer for the model
            output_dir: Directory to save the model
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs per stage
            
        Returns:
            Path to the trained model
        """
        logger.info("Starting student model training with curriculum learning")
        
        stages = ["basic", "intermediate", "advanced"]
        final_model_path = None
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Train through each curriculum stage
        for i, stage in enumerate(stages):
            if stage not in datasets:
                logger.warning(f"Stage {stage} not found in datasets, skipping")
                continue
                
            dataset = datasets[stage]
            
            # Split dataset into train/eval
            dataset = dataset.train_test_split(test_size=0.1)
            
            # Adjust learning rate for later stages (decrease as we progress)
            stage_lr = learning_rate * (0.5 ** i)
            
            # Create stage-specific output directory
            stage_output_dir = Path(output_dir) / f"stage_{stage}"
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Training {stage} stage with {len(dataset['train'])} examples, lr={stage_lr}")
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(stage_output_dir),
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=stage_lr,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_dir=str(stage_output_dir / "logs"),
                logging_steps=50,
                eval_steps=100,
                evaluation_strategy="steps",
                save_steps=500,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=True,
                report_to="none"
            )
            
            # Set up trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"]
            )
            
            # Train the model
            trainer.train()
            
            # Save model after each stage
            stage_model_path = str(stage_output_dir / "final_model")
            trainer.save_model(stage_model_path)
            tokenizer.save_pretrained(stage_model_path)
            
            final_model_path = stage_model_path
            
            logger.info(f"Completed training for {stage} stage")
            
        # Save the final model
        if final_model_path:
            # Copy the final stage model to the main output directory
            final_output_dir = Path(output_dir) / "final_model"
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # If using LoRA, merge weights into base model for the final version
            if self.use_lora:
                logger.info("Merging LoRA weights into base model")
                # Load the model from the final stage
                merged_model = PeftModel.from_pretrained(
                    AutoModelForCausalLM.from_pretrained(self.target_model, torch_dtype=torch.float16),
                    final_model_path
                )
                merged_model = merged_model.merge_and_unload()
                
                # Save the merged model
                merged_model.save_pretrained(final_output_dir)
                tokenizer.save_pretrained(final_output_dir)
                final_model_path = str(final_output_dir)
            else:
                # Just copy the final stage model
                import shutil
                for item in os.listdir(final_model_path):
                    s = os.path.join(final_model_path, item)
                    d = os.path.join(final_output_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                final_model_path = str(final_output_dir)
            
            logger.info(f"Final model saved to {final_model_path}")
            
        return final_model_path
    
    def run_enhanced_pipeline(self) -> Optional[str]:
        """
        Run the complete enhanced training pipeline.
        
        Returns:
            Path to the final trained model or None if failed
        """
        try:
            # Get training parameters from domain config
            training_params = self.domain_config.get("training_parameters", {})
            batch_size = training_params.get("batch_size", 4)
            learning_rate = training_params.get("learning_rate", 5e-5)
            num_epochs = training_params.get("num_epochs", 3)
            num_qa_pairs = training_params.get("num_qa_pairs", 100)
            
            logger.info("Starting enhanced training pipeline")
            
            # Step 1: Extract knowledge and create knowledge map
            extracted_dir = extract_knowledge(
                papers_dir=str(self.papers_dir),
                output_dir=str(self.processed_data_dir)
            )
            
            knowledge_map_path = create_knowledge_map(
                extracted_data_dir=extracted_dir,
                output_dir=str(self.processed_data_dir)
            )
            
            if not knowledge_map_path:
                logger.error("Failed to create knowledge map")
                return None
                
            # Step 2: Generate QA pairs
            qa_pairs = generate_qa_pairs(
                knowledge_map_path=knowledge_map_path,
                output_dir=str(self.processed_data_dir),
                num_pairs=num_qa_pairs
            )
            
            if not qa_pairs:
                logger.error("Failed to generate QA pairs")
                return None
                
            # Step 3: Generate enhanced responses
            enhanced_qa_pairs = generate_enhanced_responses(
                qa_pairs_path=os.path.join(self.processed_data_dir, "qa_pairs.json"),
                knowledge_map_path=knowledge_map_path,
                output_dir=str(self.processed_data_dir)
            )
            
            if not enhanced_qa_pairs:
                logger.error("Failed to enhance QA responses")
                return None
                
            # Step 4: Create curriculum dataset
            curriculum_path = create_curriculum_dataset(
                qa_pairs_path=os.path.join(self.processed_data_dir, "enhanced_qa_pairs.json"),
                output_dir=str(self.processed_data_dir)
            )
            
            if not curriculum_path:
                logger.error("Failed to create curriculum dataset")
                return None
                
            # Step 5: Create training corpus
            datasets = self.create_training_corpus(curriculum_path)
            
            if not datasets:
                logger.error("Failed to create training corpus")
                return None
                
            # Step 6: Generate model and train
            model, tokenizer = self.generate_model()
            
            model_path = self.train_student_model(
                datasets=datasets,
                model=model,
                tokenizer=tokenizer,
                output_dir=str(self.models_dir / self.target_model.split("/")[-1]),
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
            
            logger.info(f"Enhanced training pipeline completed successfully")
            return model_path
            
        except Exception as e:
            logger.error(f"Error in enhanced training pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# Main function to run enhanced training
def run_enhanced_training(
    data_dir: str,
    output_dir: str,
    domain_config: str,
    target_model: str = "distilgpt2",
    use_lora: bool = True,
    lora_r: int = 8,
    device: str = "auto",
    seed: int = 42
) -> Optional[str]:
    """
    Run enhanced training for a specific domain.
    
    Args:
        data_dir: Directory containing domain data
        output_dir: Directory to save processed data and models
        domain_config: Path to domain configuration file
        target_model: Base model to fine-tune
        use_lora: Whether to use LoRA for fine-tuning
        lora_r: LoRA attention dimension
        device: Device to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Path to the trained model or None if failed
    """
    # Set random seeds for reproducibility
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    logger.info(f"Starting enhanced training with seed {seed}")
    
    trainer = EnhancedTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        domain_config=domain_config,
        target_model=target_model,
        use_lora=use_lora,
        lora_r=lora_r,
        device=device
    )
    
    return trainer.run_enhanced_pipeline()

def main():
    """Main function to parse arguments and run enhanced training from the command line."""
    parser = argparse.ArgumentParser(description="Enhanced Training for Domain-Specific Models")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing domain data (papers, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed data and models"
    )
    
    parser.add_argument(
        "--domain-config",
        type=str,
        required=True,
        help="Path to domain configuration file"
    )
    
    parser.add_argument(
        "--target-model",
        type=str,
        default="distilgpt2",
        help="Base model to fine-tune (default: distilgpt2)"
    )
    
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA for fine-tuning"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA attention dimension (default: 8)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training (default: auto)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Run enhanced training
    result = run_enhanced_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        domain_config=args.domain_config,
        target_model=args.target_model,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        device=args.device,
        seed=args.seed
    )
    
    if result:
        logger.info(f"Enhanced training completed successfully. Model saved to {result}")
        return 0
    else:
        logger.error("Enhanced training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())