#!/usr/bin/env python3
"""
Command-line interface for Domain-LLM.
"""

import os
import sys
import click
from pathlib import Path

from purpose.processor import CombinedDataProcessor
from purpose.trainer import TransformersTrainer
from purpose.inference import ModelInference, InteractiveInterface

@click.group()
def main():
    """
    Domain-LLM: Create and use domain-specific language models.
    """
    pass

@main.command()
@click.option("--data-dir", required=True, help="Directory containing the raw data files")
@click.option("--output-dir", default="data/processed", help="Directory to save processed data")
@click.option("--max-file-size", default=50.0, type=float, help="Maximum file size to process (in MB)")
def process(data_dir, output_dir, max_file_size):
    """
    Process raw data files into a format suitable for LLM training.
    """
    click.echo(f"Processing data from {data_dir} to {output_dir}")
    
    # Create the processor
    processor = CombinedDataProcessor(
        data_dir=data_dir,
        output_dir=output_dir,
        max_file_size_mb=max_file_size
    )
    
    # Process the data
    processor.process()
    
    click.echo(f"Processing complete. Output files saved to {output_dir}")

@main.command()
@click.option("--data-dir", required=True, help="Directory containing the processed data")
@click.option("--output-dir", default="models", help="Directory to save the trained model")
@click.option("--model-name", default="gpt2", help="Base model to fine-tune")
@click.option("--output-name", default="purpose_model", help="Name for the output model directory")
@click.option("--batch-size", default=4, type=int, help="Training batch size")
@click.option("--learning-rate", default=5e-5, type=float, help="Learning rate")
@click.option("--epochs", default=3, type=int, help="Number of training epochs")
@click.option("--use-lora", is_flag=True, help="Use LoRA for parameter-efficient fine-tuning")
@click.option("--corpus-file", default="purpose_corpus.txt", help="Name of the corpus file")
@click.option("--force-cpu", is_flag=True, help="Force using CPU even if GPU is available")
def train(data_dir, output_dir, model_name, output_name, batch_size, learning_rate, 
          epochs, use_lora, corpus_file, force_cpu):
    """
    Train a language model on the processed data.
    """
    click.echo(f"Training model from {data_dir}")
    
    # Create the trainer
    trainer = TransformersTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        model_output_name=output_name,
        use_lora=use_lora,
        device="cpu" if force_cpu else None
    )
    
    # Load the data
    trainer.load_data(corpus_file=corpus_file)
    
    # Train the model
    trainer.train(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs
    )
    
    click.echo(f"Training complete. Model saved to {os.path.join(output_dir, output_name)}")

@main.command()
@click.option("--model-dir", required=True, help="Directory containing the trained model")
@click.option("--prompt", help="Text prompt for non-interactive mode")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option("--qa-mode", is_flag=True, help="Run in Q&A mode (for interactive mode)")
@click.option("--max-length", default=512, type=int, help="Maximum length of generated text")
@click.option("--temperature", default=0.7, type=float, help="Temperature for text generation")
@click.option("--force-cpu", is_flag=True, help="Force using CPU even if GPU is available")
def generate(model_dir, prompt, interactive, qa_mode, max_length, temperature, force_cpu):
    """
    Generate text or answer questions with a trained model.
    """
    # Check that either interactive mode or prompt is specified
    if not interactive and not prompt:
        click.echo("Error: Either --interactive or --prompt must be specified")
        sys.exit(1)
    
    # Load the model
    click.echo(f"Loading model from {model_dir}")
    model = ModelInference(
        model_dir=model_dir,
        device="cpu" if force_cpu else None
    )
    
    if interactive:
        # Run in interactive mode
        interface = InteractiveInterface(
            model_inference=model,
            mode="qa" if qa_mode else "chat"
        )
        interface.run()
    else:
        # Generate from prompt
        click.echo("Generating response...")
        if qa_mode:
            response = model.answer_question(
                question=prompt,
                temperature=temperature
            )
            click.echo(f"\nAnswer: {response}")
        else:
            response = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            click.echo(f"\nResponse: {response}")

@main.command()
def list_models():
    """
    List available base models for fine-tuning.
    """
    click.echo("Available base models for fine-tuning:")
    click.echo("  - gpt2 (small, 124M parameters)")
    click.echo("  - gpt2-medium (medium, 355M parameters)")
    click.echo("  - gpt2-large (large, 774M parameters)")
    click.echo("  - gpt2-xl (extra-large, 1.5B parameters)")
    click.echo("  - distilgpt2 (distilled, 82M parameters)")
    click.echo("  - facebook/opt-125m (small, 125M parameters)")
    click.echo("  - facebook/opt-350m (medium, 350M parameters)")
    click.echo("  - EleutherAI/pythia-70m (small, 70M parameters)")
    click.echo("  - EleutherAI/pythia-160m (medium, 160M parameters)")
    click.echo("")
    click.echo("Note: For custom fine-tuning, smaller models are recommended")
    click.echo("unless you have significant computational resources.")

if __name__ == "__main__":
    main() 