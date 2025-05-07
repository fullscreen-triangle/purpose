"""
CLI Command Runners

This module contains the command runners for the CLI interface.
It provides functions to run various commands like training, processing, etc.
"""

import logging
from pathlib import Path
from typing import Optional

from main.cli.process import process_data
from main.cli.trainer import EnhancedDistributedTrainer

logger = logging.getLogger(__name__)

def run_train(
    model_name: str = "gpt2",
    output_name: str = "enhanced_model",
    data_dir: str = "data",
    model_dir: str = "models",
    use_lora: bool = True,
    memory_fraction: float = 0.8,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
) -> None:
    """Run the training command."""
    trainer = EnhancedDistributedTrainer(
        model_name=model_name,
        output_name=output_name,
        data_dir=data_dir,
        model_dir=model_dir,
        use_lora=use_lora,
        memory_fraction=memory_fraction
    )
    
    trainer.train(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

def run_process(
    data_dir: str,
    output_dir: str,
    max_examples: Optional[int] = None
) -> None:
    """Run the data processing command."""

    process_data(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        max_examples=max_examples
    )

