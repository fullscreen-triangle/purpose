#!/usr/bin/env python3
"""
Command-line interface for working with Ollama models in the Purpose framework.
"""

import os
import click
import logging
from pathlib import Path
from typing import Optional

from main.cli.distributed_processor import DistributedProcessor
from main.inference.kb_query import KnowledgeBaseQuery
from main.inference.llama_inference import LlamaInference
from main.pipelines.llama_pipeline import LlamaDistillationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def main():
    """
    Purpose Ollama: Tools for working with Ollama models for domain-specific tasks.
    """
    pass


@main.command()
@click.option("--papers-dir", required=True, help="Directory containing PDF papers")
@click.option("--output-dir", default="output", help="Directory to save outputs")
@click.option("--ollama-model", default="llama2", help="Ollama model to use")
@click.option("--num-qa-pairs", default=100, type=int, help="Number of QA pairs to generate")
@click.option("--batch-size", default=4, type=int, help="Batch size for processing")
@click.option("--use-dask", is_flag=True, help="Use Dask for distributed processing")
@click.option("--workers", default=4, type=int, help="Number of workers for distributed processing")
def distill(papers_dir: str, output_dir: str, ollama_model: str, num_qa_pairs: int,
            batch_size: int, use_dask: bool, workers: int):
    """
    Distill knowledge from academic papers into an Ollama model.

    This process extracts knowledge from academic papers, generates QA pairs,
    and fine-tunes an Ollama model to create a domain-specific model.
    """
    click.echo(f"Starting knowledge distillation using Ollama model: {ollama_model}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the pipeline
    pipeline = LlamaDistillationPipeline(
        ollama_model=ollama_model,
        output_dir=output_dir,
        use_dask=use_dask,
        num_workers=workers
    )

    # Generate knowledge distillation data
    qa_pairs = pipeline.generate_knowledge_distillation_data(
        papers_dir=papers_dir,
        num_qa_pairs=num_qa_pairs,
        batch_size=batch_size
    )

    click.echo(f"Generated {len(qa_pairs)} QA pairs for knowledge distillation")

    # Run the distillation process (fine-tuning Ollama model)
    model_path = pipeline.run_distillation(qa_pairs)

    click.echo(f"Knowledge distillation complete! Model available at: {model_path}")


@main.command()
@click.option("--ollama-model", required=True, help="Ollama model to use")
@click.option("--prompt", help="Text prompt for non-interactive mode")
@click.option("--system-prompt", help="System prompt for model context")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option("--temperature", default=0.7, type=float, help="Temperature for text generation")
@click.option("--max-tokens", default=512, type=int, help="Maximum tokens to generate")
def generate(ollama_model: str, prompt: Optional[str], system_prompt: Optional[str],
             interactive: bool, temperature: float, max_tokens: int):
    """
    Generate text using an Ollama model.

    Can be used in interactive mode or with a single prompt.
    """
    # Check that either interactive mode or prompt is specified
    if not interactive and not prompt:
        click.echo("Error: Either --interactive or --prompt must be specified")
        return

    # Initialize the Ollama inference module
    model = LlamaInference(
        model_path=ollama_model,
        temperature=temperature,
        max_new_tokens=max_tokens
    )

    click.echo(f"Using Ollama model: {ollama_model}")

    if interactive:
        click.echo("Starting interactive mode. Type 'exit' to quit.")
        # Interactive loop
        while True:
            user_input = input("\nPrompt> ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            response = model.generate(
                prompt=user_input,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_tokens
            )

            click.echo(f"\nResponse: {response['generated_text']}")
    else:
        # Generate from single prompt
        click.echo("Generating response...")
        response = model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        click.echo(f"\nResponse: {response['generated_text']}")


@main.command()
@click.option("--papers-dir", required=True, help="Directory containing PDF papers")
@click.option("--output-dir", default="kb_output", help="Directory to save knowledge base")
@click.option("--use-dask", is_flag=True, help="Use Dask for distributed processing")
@click.option("--use-ray", is_flag=True, help="Use Ray for distributed processing")
@click.option("--workers", default=4, type=int, help="Number of workers for distributed processing")
def build_knowledge_base(papers_dir: str, output_dir: str, use_dask: bool,
                         use_ray: bool, workers: int):
    """
    Build a knowledge base from academic papers on sprint running.

    Extracts text from PDFs, processes it, and builds a vector database
    that can be used for retrieval and question answering.
    """

    click.echo(f"Building knowledge base from papers in {papers_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the processor
    processor = DistributedProcessor(
        use_dask=use_dask,
        use_ray=use_ray,
        max_workers=workers
    )

    # Process papers and build knowledge base
    kb_path = processor.build_knowledge_base(
        papers_dir=papers_dir,
        output_dir=output_dir
    )

    click.echo(f"Knowledge base built successfully! Available at: {kb_path}")


@main.command()
@click.option("--ollama-model", required=True, help="Ollama model to use")
@click.option("--kb-path", required=True, help="Path to the knowledge base")
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option("--query", help="Query for non-interactive mode")
@click.option("--temperature", default=0.7, type=float, help="Temperature for text generation")
def query_kb(ollama_model: str, kb_path: str, interactive: bool,
             query: Optional[str], temperature: float):
    """
    Query a knowledge base using an Ollama model.

    Combines the knowledge base with the Ollama model to answer queries.
    """
    # Check that either interactive mode or query is specified
    if not interactive and not query:
        click.echo("Error: Either --interactive or --query must be specified")
        return

    click.echo(f"Initializing knowledge base query system with model: {ollama_model}")

    # Initialize query system - The implementation details would be in a separate module


    kb_query = KnowledgeBaseQuery(
        model_name=ollama_model,
        kb_path=kb_path,
        temperature=temperature
    )

    if interactive:
        click.echo("Starting interactive query mode. Type 'exit' to quit.")
        # Interactive loop
        while True:
            user_query = input("\nQuery> ")
            if user_query.lower() in ["exit", "quit", "q"]:
                break

            click.echo("Searching knowledge base...")
            answer = kb_query.query(user_query)

            click.echo(f"\nAnswer: {answer}")
    else:
        # Answer single query
        click.echo("Searching knowledge base...")
        answer = kb_query.query(query)
        click.echo(f"\nAnswer: {answer}")


def setup_llama_model(llama_path: str) -> Optional[str]:
    """
    Set up a LLaMA model for use in knowledge distillation.
    
    Args:
        llama_path: Path to the LLaMA model
        
    Returns:
        Model name to use or None if setup failed
    """
    click.echo(f"Using local LLaMA model from: {llama_path}")
    
    # Check for required libraries
    try:
        import bitsandbytes
        from transformers import LlamaForCausalLM, LlamaTokenizer
        click.echo("Found required libraries for LLaMA models")
    except ImportError as e:
        click.echo(f"Error: Missing required libraries for LLaMA models: {e}")
        click.echo("Please install: pip install bitsandbytes transformers>=4.30.0")
        return None
    
    # Return the model path to use
    return llama_path


if __name__ == "__main__":
    main()