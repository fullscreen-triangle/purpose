#!/usr/bin/env python3
"""
Command-line interface for Domain-LLM.
"""

import os
import sys
import click
from pathlib import Path

# Import CLI modules from the cli directory
from main.cli.runner import run_train, run_process, run_visualize
from main.cli.process import process_data
from main.cli.trainer import EnhancedDistributedTrainer
from main.cli.llama_cli import setup_llama_model
from main.cli.enhanced_training import run_enhanced_training

# Import configuration system
from main.utils.config import (
    load_config, get_config, 
    get_processor_config, 
    get_trainer_config, 
    get_inference_config
)

# Import DI utilities
from main.utils.services import get_processor, get_trainer, get_inference

# Import core modules for backward compatibility
from main.trainer.knowledge_distillation import run_distillation
from main.trainer.enhanced_distillation import run_enhanced_distillation

@click.group()
@click.option("--config", help="Path to configuration file")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def main(config, verbose):
    """
    Domain-LLM: Create and use domain-specific language models.
    """
    # Load configuration
    if config:
        load_config(config)
    else:
        # Try to load from default locations
        default_configs = [
            "main_config.yaml",
            "configs/default.yaml",
            os.path.expanduser("~/.main/config.yaml")
        ]
        
        for config_path in default_configs:
            if os.path.exists(config_path):
                load_config(config_path)
                break
    
    # Set verbosity
    if verbose:
        import logging
        logging.getLogger("main").setLevel(logging.DEBUG)

@main.command()
@click.option("--data-dir", help="Directory containing the raw data files")
@click.option("--output-dir", help="Directory to save processed data")
@click.option("--max-file-size", type=float, help="Maximum file size to process (in MB)")
@click.option("--distributed", is_flag=True, help="Use distributed processing for large datasets")
@click.option("--pipeline", is_flag=True, help="Use pipeline processing architecture")
def process(data_dir, output_dir, max_file_size, distributed, pipeline):
    """
    Process raw data files into a format suitable for LLM training.
    """
    # Get processor configuration
    processor_config = get_processor_config()
    
    # Override with command-line options
    if data_dir:
        processor_config.data_dir = data_dir
    if output_dir:
        processor_config.output_dir = output_dir
    if max_file_size:
        processor_config.max_file_size_mb = max_file_size
    
    click.echo(f"Processing data from {processor_config.data_dir} to {processor_config.output_dir}")
    
    if distributed:
        # Use the distributed processor from cli module
        process_data(
            processor_config.data_dir, 
            processor_config.output_dir, 
            processor_config.max_file_size_mb
        )
    elif pipeline:
        # Use the pipeline processing architecture
        from main.pipelines.processing import run_processing_pipeline
        result = run_processing_pipeline(
            data_dir=processor_config.data_dir,
            output_dir=processor_config.output_dir,
            max_file_size_mb=processor_config.max_file_size_mb
        )
        stats = result.get('stats', {})
        click.echo(f"Processed {stats.get('num_records', 0)} records")
    else:
        # Use the dependency injection to get the processor
        processor = get_processor()
        processor.config = processor_config
        processor.process()
    
    click.echo(f"Processing complete. Output files saved to {processor_config.output_dir}")

@main.command()
@click.option("--data-dir", help="Directory containing the processed data")
@click.option("--output-dir", help="Directory to save the trained model")
@click.option("--model-name", help="Base model to fine-tune")
@click.option("--output-name", help="Name for the output model directory")
@click.option("--batch-size", type=int, help="Training batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--use-lora", is_flag=True, help="Use LoRA for parameter-efficient fine-tuning")
@click.option("--corpus-file", help="Name of the corpus file")
@click.option("--force-cpu", is_flag=True, help="Force using CPU even if GPU is available")
@click.option("--distributed", is_flag=True, help="Use distributed training for large models")
def train(data_dir, output_dir, model_name, output_name, batch_size, learning_rate, 
          epochs, use_lora, corpus_file, force_cpu, distributed):
    """
    Train a language model on the processed data.
    """
    # Get trainer configuration
    trainer_config = get_trainer_config()
    
    # Override with command-line options
    if data_dir:
        trainer_config.data_dir = data_dir
    if output_dir:
        trainer_config.output_dir = output_dir
    if model_name:
        trainer_config.model_name = model_name
    if output_name:
        trainer_config.model_output_name = output_name
    if batch_size:
        trainer_config.batch_size = batch_size
    if learning_rate:
        trainer_config.learning_rate = learning_rate
    if epochs:
        trainer_config.num_epochs = epochs
    if use_lora:
        trainer_config.use_lora = use_lora
    if force_cpu:
        trainer_config.device = "cpu"
    
    click.echo(f"Training model from {trainer_config.data_dir}")
    
    if distributed:
        # Use enhanced distributed trainer from cli module
        trainer = EnhancedDistributedTrainer(
            model_name=trainer_config.model_name,
            output_name=trainer_config.model_output_name,
            data_dir=trainer_config.data_dir,
            model_dir=trainer_config.output_dir,
            use_lora=trainer_config.use_lora
        )
        
        trainer.train(
            num_epochs=trainer_config.num_epochs,
            batch_size=trainer_config.batch_size,
            learning_rate=trainer_config.learning_rate
        )
    else:
        # Use the dependency injection to get the trainer
        trainer = get_trainer()
        trainer.config = trainer_config
        
        # Load the data
        trainer.load_data(corpus_file=corpus_file or "purpose_corpus.txt")
        
        # Train the model
        training_output = trainer.train(
            batch_size=trainer_config.batch_size,
            learning_rate=trainer_config.learning_rate,
            num_epochs=trainer_config.num_epochs
        )
        
        model_path = training_output.model_path if hasattr(training_output, 'model_path') else trainer_config.output_dir
    
    click.echo(f"Training complete. Model saved to {os.path.join(trainer_config.output_dir, trainer_config.model_output_name)}")

@main.command()
@click.option("--papers-dir", required=True, help="Directory containing PDF papers")
@click.option("--data-dir", help="Directory for processed data")
@click.option("--output-dir", help="Directory to save the model")
@click.option("--model-name", help="Base model to fine-tune")
@click.option("--use-llama", is_flag=True, help="Use a local LLaMA model instead of default models")
@click.option("--llama-path", help="Path to local LLaMA model (required if use-llama is set)")
@click.option("--num-qa-pairs", type=int, help="Number of QA pairs to generate")
@click.option("--batch-size", type=int, help="Training batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--quantize", is_flag=True, help="Use quantization to reduce model size")
@click.option("--bit-precision", type=click.Choice(['4', '8', 'none']), help="Quantization precision (4-bit, 8-bit, or none)")
@click.option("--lora-r", type=int, help="LoRA attention dimension (lower = smaller model)")
def distill(papers_dir, data_dir, output_dir, model_name, use_llama, llama_path, 
           num_qa_pairs, batch_size, learning_rate, epochs, quantize, bit_precision, lora_r):
    """
    Distill knowledge from large LLMs into a small domain-specific model.
    
    Can use OpenAI/Claude for knowledge extraction and a local LLaMA model for training.
    """
    # Get trainer configuration
    trainer_config = get_trainer_config()
    
    # Override with command-line options
    if data_dir:
        processed_data_dir = data_dir
    else:
        processed_data_dir = os.path.join(get_config().data_dir, "processed")
    
    if output_dir:
        models_dir = output_dir
    else:
        models_dir = trainer_config.output_dir
    
    if model_name:
        student_model_name = model_name
    else:
        student_model_name = trainer_config.model_name
    
    if batch_size:
        trainer_config.batch_size = batch_size
    
    if learning_rate:
        trainer_config.learning_rate = learning_rate
    
    if epochs:
        trainer_config.num_epochs = epochs
    
    if lora_r:
        trainer_config.lora_r = lora_r
    
    click.echo(f"Starting knowledge distillation from papers in {papers_dir}")
    
    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Handle LLaMA models
    if use_llama:
        if not llama_path:
            click.echo("Error: --llama-path is required when --use-llama is specified")
            sys.exit(1)
        
        # Use the dedicated setup function from llama_cli module
        student_model_name = setup_llama_model(llama_path)
        if not student_model_name:
            sys.exit(1)
    
    # Handle quantization based on bit precision
    if bit_precision:
        if bit_precision == '4':
            quantize = True
            click.echo("Using 4-bit quantization for reduced memory usage")
        elif bit_precision == '8':
            quantize = True
            click.echo("Using 8-bit quantization for reduced memory usage")
        else:
            quantize = False
            click.echo("Using full precision (no quantization)")
    
    # Run distillation
    model_path = run_distillation(
        papers_dir=papers_dir,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        student_model_name=student_model_name,
        num_qa_pairs=num_qa_pairs or 100,
        batch_size=trainer_config.batch_size,
        learning_rate=trainer_config.learning_rate,
        num_epochs=trainer_config.num_epochs,
        quantize=quantize,
        lora_r=trainer_config.lora_r
    )
    
    if model_path:
        click.echo(f"Knowledge distillation complete! Model saved to {model_path}")
    else:
        click.echo("Knowledge distillation failed. Check logs for details.")
        sys.exit(1)

@main.command()
@click.option("--papers-dir", required=True, help="Directory containing PDF papers")
@click.option("--data-dir", help="Directory for processed data")
@click.option("--output-dir", help="Directory to save the model")
@click.option("--model-name", help="Base model to fine-tune")
@click.option("--use-llama", is_flag=True, help="Use a local LLaMA model instead of default models")
@click.option("--llama-path", help="Path to local LLaMA model (required if use-llama is set)")
@click.option("--num-qa-pairs", type=int, help="Number of QA pairs to generate")
@click.option("--batch-size", type=int, help="Training batch size")
@click.option("--learning-rate", type=float, help="Learning rate")
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--quantize", is_flag=True, help="Use quantization to reduce model size")
@click.option("--bit-precision", type=click.Choice(['4', '8', 'none']), help="Quantization precision (4-bit, 8-bit, or none)")
@click.option("--lora-r", type=int, help="LoRA attention dimension (lower = smaller model)")
def enhanced_distill(papers_dir, data_dir, output_dir, model_name, use_llama, llama_path, 
                    num_qa_pairs, batch_size, learning_rate, epochs, quantize, bit_precision, lora_r):
    """
    Run enhanced knowledge distillation with multi-stage approach.
    """
    # Get trainer configuration
    trainer_config = get_trainer_config()
    
    # Override with command-line options
    if data_dir:
        processed_data_dir = data_dir
    else:
        processed_data_dir = os.path.join(get_config().data_dir, "processed")
    
    if output_dir:
        models_dir = output_dir
    else:
        models_dir = trainer_config.output_dir
    
    if model_name:
        student_model_name = model_name
    else:
        student_model_name = trainer_config.model_name
    
    if batch_size:
        trainer_config.batch_size = batch_size
    
    if learning_rate:
        trainer_config.learning_rate = learning_rate
    
    if epochs:
        trainer_config.num_epochs = epochs
    
    if lora_r:
        trainer_config.lora_r = lora_r
    
    click.echo(f"Starting enhanced knowledge distillation from papers in {papers_dir}")
    
    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Handle LLaMA models
    if use_llama:
        if not llama_path:
            click.echo("Error: --llama-path is required when --use-llama is specified")
            sys.exit(1)
        
        # Use the dedicated setup function from llama_cli module
        student_model_name = setup_llama_model(llama_path)
        if not student_model_name:
            sys.exit(1)
    
    # Handle quantization based on bit precision
    if bit_precision:
        if bit_precision == '4':
            quantize = True
            click.echo("Using 4-bit quantization for reduced memory usage")
        elif bit_precision == '8':
            quantize = True
            click.echo("Using 8-bit quantization for reduced memory usage")
        else:
            quantize = False
            click.echo("Using full precision (no quantization)")
    
    # Use the enhanced training module from cli directory
    model_path = run_enhanced_training(
        papers_dir=papers_dir,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        student_model_name=student_model_name,
        num_qa_pairs=num_qa_pairs or 100,
        batch_size=trainer_config.batch_size,
        learning_rate=trainer_config.learning_rate,
        num_epochs=trainer_config.num_epochs,
        quantize=quantize,
        lora_r=trainer_config.lora_r
    )
    
    if model_path:
        click.echo(f"Enhanced knowledge distillation complete! Model saved to {model_path}")
    else:
        click.echo("Enhanced knowledge distillation failed. Check logs for details.")
        sys.exit(1)

@main.command()
@click.option("--model-dir", help="Directory containing the trained model")
@click.option("--prompt", help="Text prompt for non-interactive mode")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option("--qa-mode", is_flag=True, help="Run in Q&A mode (for interactive mode)")
@click.option("--max-length", type=int, help="Maximum length of generated text")
@click.option("--temperature", type=float, help="Temperature for text generation")
@click.option("--force-cpu", is_flag=True, help="Force using CPU even if GPU is available")
def generate(model_dir, prompt, interactive, qa_mode, max_length, temperature, force_cpu):
    """
    Generate text using a trained model.
    """
    # Get inference configuration
    inference_config = get_inference_config()
    
    # Override with command-line options
    if model_dir:
        inference_config.model_dir = model_dir
    if max_length:
        inference_config.max_length = max_length
    if temperature:
        inference_config.temperature = temperature
    if force_cpu:
        inference_config.device = "cpu"
    
    # Use the dependency injection to get the inference engine
    inference = get_inference()
    inference.config = inference_config
    
    if interactive:
        # Enter interactive mode
        from main.inference.interface import InteractiveInterface
        interface = InteractiveInterface(
            inference=inference,
            max_length=inference_config.max_length,
            temperature=inference_config.temperature,
            qa_mode=qa_mode
        )
        interface.run()
    elif prompt:
        # Single prompt processing
        response = inference.generate_text(
            prompt=prompt,
            max_length=inference_config.max_length,
            temperature=inference_config.temperature
        )
        click.echo(response)
    else:
        click.echo("Error: Either --prompt or --interactive must be specified")
        sys.exit(1)

@main.command()
def list_models():
    """
    List available pre-trained models and fine-tuned models.
    """
    # Show installed models
    models_dir = Path(get_config().models_dir)
    
    click.echo("Installed models:")
    if models_dir.exists():
        models = [d.name for d in models_dir.iterdir() if d.is_dir()]
        for model in models:
            click.echo(f" - {model}")
    else:
        click.echo(" No local models found")
    
    # Show popular HF models
    click.echo("\nPopular HuggingFace models that can be used:")
    popular_models = [
        "gpt2", "gpt2-medium", "distilgpt2", "facebook/opt-125m", 
        "facebook/opt-350m", "EleutherAI/pythia-70m"
    ]
    for model in popular_models:
        click.echo(f" - {model}")
    
    click.echo("\nTo use a local model, provide the path to the model directory.")

@main.command()
@click.option("--data-dir", help="Directory containing visualization data")
@click.option("--output-dir", help="Directory to save visualization outputs")
def visualize(data_dir, output_dir):
    """
    Generate visualizations from processed data.
    """
    # Get configuration
    config = get_config()
    
    # Use provided values or defaults from config
    data_dir = data_dir or os.path.join(config.data_dir, "processed")
    output_dir = output_dir or os.path.join(config.output_dir, "visualizations")
    
    # Call the run_visualize function from the cli module
    run_visualize(data_dir, output_dir)
    
    click.echo(f"Visualizations generated and saved to {output_dir}")

@main.command()
@click.argument("output_path", type=click.Path(), required=False)
def config_init(output_path):
    """
    Initialize a configuration file with default settings.
    
    If OUTPUT_PATH is not provided, the configuration will be saved to main_config.yaml
    in the current directory.
    """
    from main.utils.config import ProjectConfig
    
    # Create default configuration
    default_config = ProjectConfig()
    
    # Save to file
    output_path = output_path or "main_config.yaml"
    default_config.to_file(output_path)
    
    click.echo(f"Created configuration file: {output_path}")
    click.echo("You can edit this file to customize the project settings.")

if __name__ == "__main__":
    main() 