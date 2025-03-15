"""
CLI tool for running the World Athletics scraper.
"""

import os
import logging
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.logging import RichHandler

from scraper.worldathletics_scraper import first_page_scrape, second_page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("scraper-cli")

# Create Typer app
app = typer.Typer(help="World Athletics data scraper tool")
console = Console()


@app.command()
def scrape_events(
    output_dir: str = typer.Option("data/scraped", "--output", "-o", help="Directory to save scraped data"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of events to scrape")
):
    """
    Scrape event data from World Athletics website.
    """
    console.print("[bold blue]Starting World Athletics event scraper[/bold blue]")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First page scrape - get list of events
    console.print("[yellow]Scraping event list...[/yellow]")
    events_df = first_page_scrape()
    
    if limit:
        events_df = events_df.head(limit)
    
    # Save events list
    events_path = output_path / "events.csv"
    events_df.to_csv(events_path, index=False)
    console.print(f"[green]Saved {len(events_df)} events to {events_path}[/green]")
    
    # Second page scrape - get event details and results URLs
    console.print("[yellow]Scraping event details and results URLs...[/yellow]")
    
    all_results = []
    for i, row in events_df.iterrows():
        meeting = row["Meeting"]
        url = row["URL"]
        
        console.print(f"[cyan]Processing event {i+1}/{len(events_df)}: {meeting}[/cyan]")
        
        # Create results directory for this event
        event_dir = output_path / f"event_{i}"
        event_dir.mkdir(exist_ok=True)
        
        # Get results URLs
        results_file = event_dir / "results_urls.csv"
        results_df = second_page(url, str(results_file))
        
        if results_df is not None and not results_df.empty:
            console.print(f"[green]Found {len(results_df)} results for {meeting}[/green]")
            all_results.append(results_df)
        else:
            console.print(f"[red]No results found for {meeting}[/red]")
    
    console.print("[bold green]Scraping complete![/bold green]")


@app.command()
def extract_sprint_data(
    input_dir: str = typer.Option("data/scraped", "--input", "-i", help="Directory with scraped data"),
    output_file: str = typer.Option("data/sprint_data.csv", "--output", "-o", help="Output CSV file")
):
    """
    Extract sprint running data from scraped results pages.
    """
    console.print("[bold blue]Starting sprint data extraction[/bold blue]")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]Input directory {input_dir} does not exist[/red]")
        raise typer.Exit(1)
    
    # Find all event directories
    event_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("event_")]
    
    if not event_dirs:
        console.print(f"[red]No event directories found in {input_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[yellow]Found {len(event_dirs)} event directories[/yellow]")
    
    # Process each event
    all_sprint_data = []
    
    for event_dir in event_dirs:
        # Load results URLs
        results_file = event_dir / "results_urls.csv"
        if not results_file.exists():
            console.print(f"[red]No results file found in {event_dir}[/red]")
            continue
        
        try:
            results_df = pd.read_csv(results_file)
            
            # Filter for sprint events (100m, 200m, 400m)
            sprint_events = results_df[
                results_df["Event"].str.contains("100m|200m|400m", case=False, regex=True)
            ]
            
            if sprint_events.empty:
                console.print(f"[yellow]No sprint events found in {event_dir}[/yellow]")
                continue
            
            console.print(f"[cyan]Processing {len(sprint_events)} sprint events from {event_dir.name}[/cyan]")
            
            # Extract data from each sprint event page
            for _, row in sprint_events.iterrows():
                event_name = row["Event"]
                url = row["URL"]
                
                try:
                    # Extract data using the base scraper's extract_data_from_page function
                    event_data = extract_data_from_page(url)
                    
                    if event_data:
                        all_sprint_data.extend(event_data)
                        console.print(f"[green]Extracted data for {event_name}[/green]")
                    else:
                        console.print(f"[red]Failed to extract data for {event_name}[/red]")
                
                except Exception as e:
                    console.print(f"[red]Error extracting data from {url}: {str(e)}[/red]")
        
        except Exception as e:
            console.print(f"[red]Error processing {results_file}: {str(e)}[/red]")
    
    # Save all sprint data to CSV
    if all_sprint_data:
        import pandas as pd
        df = pd.DataFrame(all_sprint_data)
        df.to_csv(output_file, index=False)
        console.print(f"[bold green]Saved {len(df)} sprint data records to {output_file}[/bold green]")
    else:
        console.print("[red]No sprint data was extracted[/red]")


@app.command("run")
def run_scraper(
    output_dir: str = typer.Option("./scraped_data", help="Directory to save scraped data"),
    verify_ssl: bool = typer.Option(True, help="Enable/disable SSL certificate verification")
):
    """
    Run the World Athletics scraper to collect sprint data.
    """
    console = Console()
    
    console.print("[bold blue]Running World Athletics Scraper...[/bold blue]")
    
    try:
        from scraper.worldathletics_scraper import scrape_results
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Change working directory temporarily
        original_dir = os.getcwd()
        os.chdir(output_dir)
        
        console.print(f"[yellow]Saving results to: {output_dir}[/yellow]")
        
        # Run the scraper with SSL verification option
        results = scrape_results(verify_ssl=verify_ssl)
        
        # Return to original directory
        os.chdir(original_dir)
        
        console.print(f"[bold green]✓ Scraping complete! Data saved to {output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        if "SSL" in str(e):
            console.print("[yellow]Try running with --no-verify-ssl to bypass SSL certificate verification[/yellow]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app() 