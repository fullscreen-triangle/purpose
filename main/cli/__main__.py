#!/usr/bin/env python3
"""
Main entry point for the main.cli package.
This module allows the package to be executed directly using python -m main.cli
"""

from main.cli import app
from main.cli.distillation_cli import run_distillation_pipeline
import typer

# Add the enhanced_distill command to the main app
@app.command()
def enhanced_distill(
    papers_dir: str = typer.Argument(..., help="Directory containing PDF papers"),
    model_name: str = typer.Option("distilgpt2", help="Base model to fine-tune"),
    output_dir: str = typer.Option(None, help="Directory to save models"),
    data_dir: str = typer.Option(None, help="Directory for processed data"),
    num_qa_pairs: int = typer.Option(100, help="Number of QA pairs to generate"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate for training"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    quantize: bool = typer.Option(False, help="Use quantization to reduce model size"),
    lora_r: int = typer.Option(4, help="LoRA attention dimension"),
    seed: int = typer.Option(42, help="Random seed for reproducibility")
):
    """Run the enhanced knowledge distillation pipeline."""
    result = run_distillation_pipeline(
        papers_dir=papers_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        num_qa_pairs=num_qa_pairs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        quantize=quantize,
        lora_r=lora_r,
        seed=seed
    )
    
    if result:
        typer.echo(f"Distillation completed successfully. Model saved to: {result}")
    else:
        typer.echo("Distillation failed. Check logs for details.", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 