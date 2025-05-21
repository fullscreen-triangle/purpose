"""
Command-line interface for the Purpose Model Hub.

This module provides CLI commands to list, search, and use specialized models
from the model hub.
"""

import os
import sys
import json
import typer
import asyncio
from enum import Enum
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

from main.utils.model_hub import ModelHub, PurposeAPIClient, ModelSource, TaskType, ModelInfo

# Create Typer app
app = typer.Typer(help="Commands for working with the Purpose Model Hub")
console = Console()

# Get the API token from environment variables
DEFAULT_API_TOKEN = os.environ.get("HUGGINGFACE_API_KEY", "")
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.purpose/model_hub_config.json")

@app.callback()
def callback():
    """
    Purpose Model Hub: Access specialized AI models for different tasks.
    """
    pass

@app.command()
def list_models(
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Filter models by task type"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter models by source"),
    context_window: Optional[int] = typer.Option(None, "--context", "-c", help="Minimum context window size"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-f", help="Path to config file")
):
    """
    List available models in the hub.
    """
    try:
        # Initialize the model hub
        model_hub = ModelHub(str(config_path) if config_path else None)
        
        # Create a table
        table = Table(title="Available Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Size", style="blue")
        table.add_column("Context", style="magenta")
        table.add_column("Specialties", style="yellow")
        
        # Filter models based on options
        filtered_models = []
        for model_id, model_info in model_hub.models.items():
            # Filter by task if specified
            if task:
                try:
                    task_enum = TaskType(task)
                    if task_enum not in model_info.specialties:
                        continue
                except ValueError:
                    console.print(f"[yellow]Warning: Unknown task type '{task}'[/yellow]")
                    # Still include the model in case we're doing a string match
                    if not any(task.lower() in specialty.value.lower() for specialty in model_info.specialties):
                        continue
            
            # Filter by source if specified
            if source and source.lower() != model_info.source.value.lower():
                continue
            
            # Filter by context window if specified
            if context_window and model_info.context_window < context_window:
                continue
            
            filtered_models.append(model_info)
        
        # Add rows to table
        for model_info in filtered_models:
            specialties = ", ".join(s.value for s in model_info.specialties)
            table.add_row(
                model_info.model_id,
                model_info.source.value,
                model_info.size_params,
                str(model_info.context_window),
                specialties
            )
        
        # Print the table
        console.print(table)
        console.print(f"Found {len(filtered_models)} models matching criteria")
        
    except Exception as e:
        console.print(f"[bold red]Error listing models: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def list_tasks():
    """
    List available task types.
    """
    try:
        # Create a table
        table = Table(title="Available Task Types")
        table.add_column("Task Type", style="cyan")
        table.add_column("Description", style="green")
        
        # Task descriptions
        task_descriptions = {
            TaskType.BASE_TRAINING: "Models for base model training",
            TaskType.DISTILLATION_TARGET: "Smaller models for knowledge distillation",
            TaskType.DATA_PROCESSING: "Models specialized in processing raw data",
            TaskType.KNOWLEDGE_MAPPING: "Models for mapping knowledge into structured formats",
            TaskType.KNOWLEDGE_EXTRACTION: "Models for extracting knowledge from unstructured text",
            TaskType.QUERY_GENERATION: "Models for generating queries based on knowledge",
            TaskType.RESPONSE_GENERATION: "Models for generating detailed responses",
            TaskType.CURRICULUM_LEARNING: "Models for curriculum-based learning progression",
            TaskType.INFERENCE: "Models optimized for inference/deployment",
            TaskType.TEXT_EMBEDDING: "Models for generating text embeddings",
            TaskType.TEXT_CLASSIFICATION: "Models for classifying text",
            TaskType.REASONING: "Models with strong reasoning capabilities",
            TaskType.INSTRUCTION_FOLLOWING: "Models specialized in following instructions",
            TaskType.CODE_GENERATION: "Models for generating code",
            TaskType.MULTILINGUAL: "Models with multilingual capabilities",
        }
        
        # Add rows to table
        for task_type, description in task_descriptions.items():
            table.add_row(task_type.value, description)
        
        # Print the table
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error listing task types: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def model_info(
    model_id: str = typer.Argument(..., help="ID of the model to get information for"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-f", help="Path to config file")
):
    """
    Get detailed information about a specific model.
    """
    try:
        # Initialize the model hub
        model_hub = ModelHub(str(config_path) if config_path else None)
        
        # Get model info
        model_info = model_hub.get_model_info(model_id)
        if not model_info:
            console.print(f"[bold red]Model '{model_id}' not found in the hub[/bold red]")
            raise typer.Exit(code=1)
        
        # Print model info
        console.print(Panel.fit(
            f"[bold cyan]Model ID:[/bold cyan] {model_info.model_id}\n"
            f"[bold green]Source:[/bold green] {model_info.source.value}\n"
            f"[bold blue]Size:[/bold blue] {model_info.size_params}\n"
            f"[bold magenta]Context Window:[/bold magenta] {model_info.context_window} tokens\n"
            f"[bold yellow]Specialties:[/bold yellow] {', '.join(s.value for s in model_info.specialties)}\n"
            f"[bold white]Strengths:[/bold white]\n" + "\n".join(f"- {s}" for s in model_info.strengths),
            title=f"Model Information: {model_id}"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error getting model info: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def recommend_models(
    task: str = typer.Argument(..., help="Task type to get recommendations for"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-f", help="Path to config file")
):
    """
    Get recommended models for a specific task.
    """
    try:
        # Initialize the model hub
        model_hub = ModelHub(str(config_path) if config_path else None)
        
        # Get task recommendations
        try:
            task_enum = TaskType(task)
        except ValueError:
            console.print(f"[bold yellow]Warning: '{task}' is not a standard task type[/bold yellow]")
            # Try using the string directly
            task_enum = task
        
        recommended_models = model_hub.get_recommended_models(task_enum)
        
        if not recommended_models:
            console.print(f"[bold yellow]No recommended models found for task '{task}'[/bold yellow]")
            raise typer.Exit(code=1)
        
        # Create a table
        table = Table(title=f"Recommended Models for {task}")
        table.add_column("Model ID", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Size", style="blue")
        table.add_column("Context", style="magenta")
        
        # Add rows to table
        for model_id in recommended_models:
            model_info = model_hub.get_model_info(model_id)
            if model_info:
                table.add_row(
                    model_id,
                    model_info.source.value,
                    model_info.size_params,
                    str(model_info.context_window)
                )
        
        # Print the table
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error getting recommendations: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def process_text(
    task: str = typer.Argument(..., help="Task type to process with"),
    text: str = typer.Argument(..., help="Text to process"),
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model ID to use"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature for generation"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-n", help="Maximum number of tokens to generate"),
    api_token: str = typer.Option(DEFAULT_API_TOKEN, "--api-token", "-a", help="API token for authentication"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-f", help="Path to config file")
):
    """
    Process text using a specialized model.
    """
    try:
        if not api_token:
            console.print("[bold red]API token is required. Please provide it with --api-token or set the HUGGINGFACE_API_KEY environment variable.[/bold red]")
            raise typer.Exit(code=1)
        
        # Initialize the API client
        client = PurposeAPIClient(api_token=api_token, config_path=str(config_path) if config_path else None)
        
        # Process the text
        with Progress() as progress:
            task_progress = progress.add_task("[cyan]Processing text...", total=1)
            
            # Create and run the async function
            async def process():
                try:
                    result = await client.process_task(
                        task_type=task,
                        input_text=text,
                        model_id=model_id,
                        parameters={
                            "temperature": temperature,
                            "max_new_tokens": max_tokens
                        }
                    )
                    return result
                finally:
                    await client.close()
            
            result = asyncio.run(process())
            progress.update(task_progress, completed=1)
        
        # Print the result
        if isinstance(result, dict) and "generated_text" in result:
            console.print(Panel(result["generated_text"], title="Generated Text", border_style="green"))
        elif isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            console.print(Panel(result[0]["generated_text"], title="Generated Text", border_style="green"))
        else:
            console.print(Panel(str(result), title="Model Output", border_style="green"))
        
    except Exception as e:
        console.print(f"[bold red]Error processing text: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def setup_config(
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-f", help="Path to config file")
):
    """
    Set up configuration for the model hub.
    """
    try:
        # Create config directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # Get API keys
        api_keys = {}
        
        for source in ModelSource:
            api_key = typer.prompt(
                f"Enter API key for {source.value} (leave empty to skip)",
                default="",
                hide_input=True,
                show_default=False
            )
            if api_key:
                api_keys[source.value] = api_key
        
        # Save config
        config = {
            "api_keys": api_keys
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(f"[bold green]Configuration saved to {config_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error setting up config: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app() 