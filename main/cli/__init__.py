#!/usr/bin/env python3

from purpose.cli.runner import (
    run_train,
    run_process,
    run_visualize
)

# Import the model optimization and knowledge distillation modules
from purpose.examples.sprint.knowledge_distill import KnowledgeDistiller
from purpose.examples.sprint.model_optimization import ModelOptimizer

__all__ = [
    'run_train',
    'run_process',
    'run_visualize',
    'KnowledgeDistiller',
    'ModelOptimizer'
]

import typer
from typing import Optional
from importlib.util import find_spec

# Import CLI modules
from purpose.cli.model_hub_commands import app as model_hub_app

# Create main app
app = typer.Typer(help="Purpose CLI: A toolkit for knowledge and LLM training")

# Add subcommands
app.add_typer(model_hub_app, name="models", help="Work with the Purpose Model Hub")

# Check for optional modules and add their CLIs if available
try:
    if find_spec("purpose.cli.knowledge_commands"):
        from purpose.cli.knowledge_commands import app as knowledge_app
        app.add_typer(knowledge_app, name="knowledge", help="Work with knowledge graphs")
except ImportError:
    pass

try:
    if find_spec("purpose.cli.training_commands"):
        from purpose.cli.training_commands import app as training_app
        app.add_typer(training_app, name="train", help="Train and fine-tune models")
except ImportError:
    pass

try:
    if find_spec("purpose.cli.scraper_commands"):
        from purpose.cli.scraper_commands import app as scraper_app
        app.add_typer(scraper_app, name="scrape", help="Scrape and process content")
except ImportError:
    pass

@app.callback()
def callback():
    """
    Purpose: A toolkit for knowledge-powered LLM workflows.
    """
    pass

@app.command()
def version():
    """
    Show the Purpose version.
    """
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("purpose").version
        typer.echo(f"Purpose v{version}")
    except:
        typer.echo("Purpose (development version)")

if __name__ == "__main__":
    app() 