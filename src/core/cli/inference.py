"""
CLI tool for making inferences with the trained sprint domain LLMs.
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import typer
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.markdown import Markdown
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("inference-cli")

# Create Typer app
app = typer.Typer(help="Sprint LLM inference tool")
console = Console()


def load_model(model_path: str, device: Optional[str] = None):
    """
    Load a trained model from the specified path.
    
    Args:
        model_path: Path to the trained model directory
        device: Device to load the model on (cpu, cuda, auto)
        
    Returns:
        Loaded model and tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path {model_path} does not exist")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if this is a PEFT model
    if (model_path / "adapter_config.json").exists():
        # Load configuration to determine model type
        with open(model_path / "adapter_config.json", "r") as f:
            config = json.load(f)
        
        task_type = config.get("task_type", "")
        
        # Load base model for the adapter
        base_model_path = config.get("base_model_name_or_path", "distilbert-base-uncased")
        
        if "CAUSAL_LM" in task_type:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        else:
            base_model = AutoModelForMaskedLM.from_pretrained(
                base_model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        # Load the PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Try to load as a regular model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except:
            model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
    
    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer


@app.command()
def list_models(
    models_dir: str = typer.Option("models", "--models-dir", help="Directory containing trained models")
):
    """
    List all available trained models.
    """
    console.print("[bold blue]Available Sprint LLMs:[/bold blue]")
    
    models_path = Path(models_dir)
    if not models_path.exists():
        console.print(f"[red]Models directory {models_dir} does not exist[/red]")
        raise typer.Exit(1)
    
    # Find all model directories
    model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("sprint-llm-")]
    
    if not model_dirs:
        console.print("[yellow]No trained models found[/yellow]")
        console.print("Use the 'train' command to train a new model first")
        return
    
    # Sort models by creation time (newest first)
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Display model information
    for i, model_dir in enumerate(model_dirs):
        creation_time = model_dir.stat().st_mtime
        from datetime import datetime
        creation_date = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this is the most recent model
        is_latest = i == 0
        
        # Try to get training metrics if available
        metrics_file = model_dir / "trainer_state.json"
        metric_text = ""
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    if "best_metric" in metrics:
                        metric_text = f" (Best loss: {metrics['best_metric']:.4f})"
            except:
                pass
        
        # Display model info
        model_indicator = "[bold green]LATEST[/bold green] " if is_latest else ""
        console.print(f"{i+1}. {model_indicator}[cyan]{model_dir.name}[/cyan]")
        console.print(f"   Created: {creation_date}{metric_text}")
        console.print(f"   Path: {model_dir}")
        console.print()


@app.command()
def answer_question(
    question: str = typer.Argument(..., help="Question about sprint running to answer"),
    model_path: str = typer.Option(None, "--model", "-m", help="Path to specific model to use (will use latest if not specified)"),
    models_dir: str = typer.Option("models", "--models-dir", help="Directory containing trained models"),
    max_length: int = typer.Option(500, "--max-length", help="Maximum response length"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature (higher = more creative)")
):
    """
    Answer a sprint running related question using the trained domain model.
    """
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    console.print("[yellow]Thinking...[/yellow]")
    
    # Find the model to use
    if model_path is None:
        models_path = Path(models_dir)
        if not models_path.exists():
            console.print(f"[red]Models directory {models_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        # Find all model directories
        model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("sprint-llm-")]
        
        if not model_dirs:
            console.print("[red]No trained models found[/red]")
            console.print("Use the 'train' command to train a new model first")
            raise typer.Exit(1)
        
        # Use the newest model
        model_path = str(max(model_dirs, key=lambda x: x.stat().st_mtime))
    
    try:
        # Load the model
        model, tokenizer = load_model(model_path)
        
        # Create a pipeline for text generation
        if hasattr(model, "generate"):
            # For causal language models (like GPT)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True
            )
            
            # Generate response
            context = f"Question: {question}\n\nAnswer:"
            response = pipe(context, num_return_sequences=1)[0]["generated_text"]
            
            # Extract just the answer part
            answer = response.split("Answer:")[1].strip()
        else:
            # For masked language models (like BERT)
            pipe = pipeline(
                "fill-mask",
                model=model,
                tokenizer=tokenizer
            )
            
            # For masked models, we'll generate by iteratively filling in masks
            answer = f"Regarding the question about sprint running: {question}\n\n"
            
            # Add some domain-specific prompts
            answer += "Based on the scientific literature on sprint running, "
            
            # This is a simplistic approach - masked models aren't ideal for this type of generation
            mask_token = tokenizer.mask_token
            for _ in range(10):  # Generate 10 tokens
                masked_text = answer + mask_token
                predictions = pipe(masked_text)
                next_token = predictions[0]["token_str"]
                answer += next_token + " "
                
                if next_token in [".", "!", "?"]:
                    break
        
        # Display the answer
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(Panel(Markdown(answer), border_style="green"))
    
    except Exception as e:
        console.print(f"[bold red]Error generating answer: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def interactive_mode(
    model_path: str = typer.Option(None, "--model", "-m", help="Path to specific model to use (will use latest if not specified)"),
    models_dir: str = typer.Option("models", "--models-dir", help="Directory containing trained models"),
    max_length: int = typer.Option(500, "--max-length", help="Maximum response length"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature (higher = more creative)")
):
    """
    Start an interactive session with the sprint domain model.
    """
    console.print("[bold blue]Sprint Running Domain Expert Interactive Mode[/bold blue]")
    console.print("Type your questions about sprint running and the model will answer.")
    console.print("Type 'exit', 'quit', or Ctrl+C to end the session.\n")
    
    # Find the model to use
    if model_path is None:
        models_path = Path(models_dir)
        if not models_path.exists():
            console.print(f"[red]Models directory {models_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        # Find all model directories
        model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("sprint-llm-")]
        
        if not model_dirs:
            console.print("[red]No trained models found[/red]")
            console.print("Use the 'train' command to train a new model first")
            raise typer.Exit(1)
        
        # Use the newest model
        model_path = str(max(model_dirs, key=lambda x: x.stat().st_mtime))
    
    try:
        # Load the model
        console.print(f"[yellow]Loading model from {model_path}...[/yellow]")
        model, tokenizer = load_model(model_path)
        
        # Create a pipeline for text generation
        if hasattr(model, "generate"):
            # For causal language models (like GPT)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True
            )
            
            # Interactive loop
            while True:
                # Get user question
                question = console.input("\n[bold cyan]User:[/bold cyan] ")
                
                if question.lower() in ["exit", "quit", "q"]:
                    break
                
                # Generate response
                console.print("[yellow]Thinking...[/yellow]")
                context = f"Question: {question}\n\nAnswer:"
                response = pipe(context, num_return_sequences=1)[0]["generated_text"]
                
                # Extract just the answer part
                answer = response.split("Answer:")[1].strip()
                
                # Display the answer
                console.print("\n[bold green]Sprint Expert:[/bold green]")
                console.print(Panel(answer, border_style="green"))
        else:
            # For masked language models (like BERT)
            pipe = pipeline(
                "fill-mask",
                model=model,
                tokenizer=tokenizer
            )
            
            # Interactive loop
            while True:
                # Get user question
                question = console.input("\n[bold cyan]User:[/bold cyan] ")
                
                if question.lower() in ["exit", "quit", "q"]:
                    break
                
                # Generate response (simplified for masked models)
                console.print("[yellow]Thinking...[/yellow]")
                answer = f"Regarding your question about sprint running: {question}\n\n"
                answer += "Based on the scientific literature, "
                
                # Generate a few tokens
                mask_token = tokenizer.mask_token
                for _ in range(10):
                    masked_text = answer + mask_token
                    predictions = pipe(masked_text)
                    next_token = predictions[0]["token_str"]
                    answer += next_token + " "
                    
                    if next_token in [".", "!", "?"]:
                        break
                
                # Display the answer
                console.print("\n[bold green]Sprint Expert:[/bold green]")
                console.print(Panel(answer, border_style="green"))
    
    except Exception as e:
        console.print(f"[bold red]Error in interactive mode: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def analyze_document(
    document_path: str = typer.Argument(..., help="Path to a document to analyze"),
    model_path: str = typer.Option(None, "--model", "-m", help="Path to specific model to use (will use latest if not specified)"),
    models_dir: str = typer.Option("models", "--models-dir", help="Directory containing trained models"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format (text, json, markdown)")
):
    """
    Analyze a sprint-related document using the domain model.
    """
    document_path = Path(document_path)
    if not document_path.exists():
        console.print(f"[red]Document {document_path} does not exist[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Analyzing document:[/bold blue] {document_path}")
    
    # Read document
    try:
        with open(document_path, "r", encoding="utf-8") as f:
            document_text = f.read()
    except Exception as e:
        console.print(f"[red]Error reading document: {str(e)}[/red]")
        raise typer.Exit(1)
    
    # Find the model to use
    if model_path is None:
        models_path = Path(models_dir)
        if not models_path.exists():
            console.print(f"[red]Models directory {models_dir} does not exist[/red]")
            raise typer.Exit(1)
        
        # Find all model directories
        model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("sprint-llm-")]
        
        if not model_dirs:
            console.print("[red]No trained models found[/red]")
            console.print("Use the 'train' command to train a new model first")
            raise typer.Exit(1)
        
        # Use the newest model
        model_path = str(max(model_dirs, key=lambda x: x.stat().st_mtime))
    
    try:
        # Load the model
        model, tokenizer = load_model(model_path)
        
        # Create a pipeline for text generation
        if hasattr(model, "generate"):
            # For causal language models (like GPT)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            # Generate analysis
            prompt = (
                f"Document to analyze:\n\n{document_text[:2000]}...\n\n"
                "Please provide a technical analysis of this sprint running document, "
                "including key insights, methodology assessment, and relation to existing literature."
            )
            
            console.print("[yellow]Generating analysis...[/yellow]")
            response = pipe(prompt, num_return_sequences=1)[0]["generated_text"]
            
            # Extract just the analysis part (after the prompt)
            analysis = response.split("existing literature.")[-1].strip()
            
            # Format the output
            if output_format == "json":
                analysis_json = {
                    "document": document_path.name,
                    "model": model_path,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                console.print(json.dumps(analysis_json, indent=2))
            elif output_format == "markdown":
                markdown = f"""# Analysis of {document_path.name}

## Generated by Sprint Domain Expert LLM

{analysis}

---
*Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
                console.print(Markdown(markdown))
            else:
                # Default text format
                console.print("\n[bold green]Analysis:[/bold green]")
                console.print(Panel(analysis, border_style="green"))
        else:
            console.print("[red]Document analysis requires a generative (causal) language model[/red]")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error analyzing document: {str(e)}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 