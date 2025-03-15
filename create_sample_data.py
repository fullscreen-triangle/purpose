#!/usr/bin/env python3
"""
Create sample sprint results data for testing the data processing and model training pipeline.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sample-data-generator")

# Constants
OUTPUT_DIR = "data/sprint_data"
ATHLETES = [
    {"name": "Usain Bolt", "country": "Jamaica"},
    {"name": "Justin Gatlin", "country": "United States"},
    {"name": "Yohan Blake", "country": "Jamaica"},
    {"name": "Tyson Gay", "country": "United States"},
    {"name": "Asafa Powell", "country": "Jamaica"},
    {"name": "Noah Lyles", "country": "United States"},
    {"name": "Christian Coleman", "country": "United States"},
    {"name": "Fred Kerley", "country": "United States"},
    {"name": "Andre De Grasse", "country": "Canada"},
    {"name": "Trayvon Bromell", "country": "United States"},
    {"name": "Ferdinand Omanyala", "country": "Kenya"},
    {"name": "Su Bingtian", "country": "China"},
    {"name": "Wayde van Niekerk", "country": "South Africa"},
    {"name": "Michael Johnson", "country": "United States"},
    {"name": "LaShawn Merritt", "country": "United States"},
    {"name": "Kirani James", "country": "Grenada"},
    {"name": "Steven Gardiner", "country": "Bahamas"},
    {"name": "Michael Norman", "country": "United States"},
    {"name": "Isaac Makwala", "country": "Botswana"},
    {"name": "Akani Simbine", "country": "South Africa"}
]

COMPETITIONS = [
    "World Athletics Championships",
    "Olympic Games",
    "Diamond League Paris",
    "Diamond League Zurich",
    "Commonwealth Games",
    "European Athletics Championships",
    "Pan American Games",
    "African Championships",
    "Asian Games",
    "World Athletics Relays"
]

EVENT_TYPES = ["Men's 100m", "Men's 200m", "Men's 400m"]

# Time ranges (seconds) for each event type
TIME_RANGES = {
    "Men's 100m": (9.58, 10.5),
    "Men's 200m": (19.19, 20.8),
    "Men's 400m": (43.03, 45.5)
}

def generate_random_time(event_type):
    """Generate a random time for a given event type."""
    min_time, max_time = TIME_RANGES[event_type]
    time_seconds = random.uniform(min_time, max_time)
    
    # Format the time
    if event_type in ["Men's 100m", "Men's 200m"]:
        return f"{time_seconds:.2f}"
    else:
        # For 400m, use min:sec.ms format
        seconds = int(time_seconds)
        milliseconds = int((time_seconds - seconds) * 100)
        return f"{seconds}.{milliseconds:02d}"

def generate_competition_data(competition, event_type, date):
    """Generate results for a single competition and event."""
    # Select 8 random athletes
    selected_athletes = random.sample(ATHLETES, 8)
    
    # Generate times for each athlete
    times = [generate_random_time(event_type) for _ in range(8)]
    
    # Sort athletes by time
    athlete_times = list(zip(selected_athletes, times))
    athlete_times.sort(key=lambda x: float(x[1]))
    
    # Create results
    results = []
    for position, (athlete, time) in enumerate(athlete_times, 1):
        results.append({
            "Event": event_type,
            "Competition": competition,
            "Position": str(position),
            "Athlete": athlete["name"],
            "Country": athlete["country"],
            "Time": time,
            "Date": date.strftime("%Y-%m-%d")
        })
    
    return results

def generate_sample_data(num_competitions=30):
    """Generate sample sprint results data."""
    all_results = []
    start_date = datetime(2021, 1, 1)
    
    for i in range(num_competitions):
        # Pick a random competition
        competition = random.choice(COMPETITIONS)
        
        # Generate a random date
        days_offset = random.randint(0, 730)  # Within 2 years
        competition_date = start_date + timedelta(days=days_offset)
        
        # For each competition, generate results for each event type
        for event_type in EVENT_TYPES:
            results = generate_competition_data(competition, event_type, competition_date)
            all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df

def save_data(df, output_dir=OUTPUT_DIR):
    """Save the generated data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined results
    output_path = os.path.join(output_dir, "all_sprint_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} sample sprint results to {output_path}")
    
    # Save individual event files
    for event_type in EVENT_TYPES:
        event_df = df[df["Event"] == event_type]
        event_path = os.path.join(output_dir, f"{event_type.replace(' ', '_').replace('\'', '')}.csv")
        event_df.to_csv(event_path, index=False)
        logger.info(f"Saved {len(event_df)} {event_type} results to {event_path}")

def main():
    """Main function to generate and save sample data."""
    logger.info("Generating sample sprint results data...")
    df = generate_sample_data(num_competitions=30)
    save_data(df)
    logger.info("Sample data generation complete!")

if __name__ == "__main__":
    main() 