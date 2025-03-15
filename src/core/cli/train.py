"""
CLI tool for training language models on sprint domain data.
"""

import os
import logging
from pathlib import Path
import time

import typer
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from core.data_processing import PDFProcessor
from core.model_training import ModelTrainer
from core.continuous_learning import ContinuousLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("train-cli")

# Create Typer app
app = typer.Typer(help="Sprint LLM training tool")
console = Console()


@app.command()
def process_papers(
    papers_dir: str = typer.Option("papers", "--papers-dir", help="Directory containing PDF papers"),
    output_dir: str = typer.Option("data/processed", "--output-dir", help="Directory to save processed data"),
    max_workers: int = typer.Option(4, "--max-workers", help="Maximum number of parallel workers")
):
    """
    Process academic papers and extract text for model training.
    """
    console.print("[bold blue]Starting academic paper processing[/bold blue]")
    
    try:
        # Create processor
        processor = PDFProcessor(papers_dir=papers_dir)
        
        # Process papers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Processing papers...", total=len(processor.pdf_files))
            
            def progress_callback(completed):
                progress.update(task, completed=completed)
            
            # Save extracted texts
            processor.save_extracted_texts(output_dir=output_dir)
            progress.update(task, completed=len(processor.pdf_files))
        
        console.print(f"[bold green]Paper processing complete! Texts saved to {output_dir}[/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]Error processing papers: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def train_model(
    processed_data_dir: str = typer.Option("data/processed", "--data-dir", help="Directory with processed text data"),
    model_dir: str = typer.Option("models", "--model-dir", help="Directory to save trained models"),
    model_name: str = typer.Option("distilbert-base-uncased", "--model-name", help="Base model checkpoint to use"),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", help="Learning rate"),
    num_epochs: int = typer.Option(3, "--epochs", help="Number of training epochs"),
    use_lora: bool = typer.Option(True, "--use-lora/--no-lora", help="Whether to use LoRA for efficient fine-tuning")
):
    """
    Train a language model on processed sprint domain data.
    """
    console.print("[bold blue]Starting model training[/bold blue]")
    
    try:
        # Create trainer
        trainer = ModelTrainer(
            processed_data_dir=processed_data_dir,
            model_dir=model_dir,
            model_name=model_name,
            use_lora=use_lora
        )
        
        # Start a spinner during training
        with console.status("[yellow]Training model...[/yellow]", spinner="dots") as status:
            start_time = time.time()
            
            # Train model
            model, metrics = trainer.train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
            
            training_time = time.time() - start_time
        
        # Print training results
        console.print(f"[bold green]Training complete in {training_time:.2f} seconds![/bold green]")
        console.print("[cyan]Training metrics:[/cyan]")
        for k, v in metrics.items():
            console.print(f"  {k}: {v}")
    
    except Exception as e:
        console.print(f"[bold red]Error training model: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def continuous_learning(
    papers_dir: str = typer.Option("papers", "--papers-dir", help="Directory containing PDF papers"),
    processed_data_dir: str = typer.Option("data/processed", "--data-dir", help="Directory with processed text data"),
    model_dir: str = typer.Option("models", "--model-dir", help="Directory with saved models"),
    force_update: bool = typer.Option(False, "--force", help="Force model update even if no new papers"),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", help="Learning rate"),
    num_epochs: int = typer.Option(3, "--epochs", help="Number of training epochs")
):
    """
    Run a continuous learning cycle to process new papers and update models.
    """
    console.print("[bold blue]Starting continuous learning cycle[/bold blue]")
    
    try:
        # Create continuous learner
        learner = ContinuousLearner(
            papers_dir=papers_dir,
            processed_data_dir=processed_data_dir,
            models_dir=model_dir
        )
        
        # Check for new papers
        console.print("[yellow]Checking for new papers...[/yellow]")
        new_papers = learner.check_for_new_papers()
        
        if new_papers:
            console.print(f"[green]Found {len(new_papers)} new papers to process[/green]")
            
            # Process new papers
            console.print("[yellow]Processing new papers...[/yellow]")
            learner.process_new_papers()
            
            # Update model
            console.print("[yellow]Updating model with new papers...[/yellow]")
            updated_model = learner.update_model(
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
            
            if updated_model:
                console.print(f"[bold green]Model updated successfully: {updated_model}[/bold green]")
            else:
                console.print("[bold red]Model update failed[/bold red]")
        
        elif force_update:
            console.print("[yellow]No new papers found, but forcing model update...[/yellow]")
            
            # Force model update
            updated_model = learner.update_model(
                force_update=True,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
            
            if updated_model:
                console.print(f"[bold green]Model updated successfully: {updated_model}[/bold green]")
            else:
                console.print("[bold red]Model update failed[/bold red]")
        
        else:
            console.print("[green]No new papers to process and force_update is False. No model update needed.[/green]")
    
    except Exception as e:
        console.print(f"[bold red]Error in continuous learning cycle: {str(e)}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 