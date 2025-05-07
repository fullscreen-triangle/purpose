#!/usr/bin/env python3
"""
Generate sample sprint data for testing the Domain-LLM framework.

This script creates a small set of CSV and JSON files with sprint-related data
that can be used to test the data processing, training, and inference pipeline.
"""

import os
import json
import random
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Constants
OUTPUT_DIR = "content"
NUM_ATHLETES = 50
NUM_RACES = 20

# Sample data
FIRST_NAMES = ["Usain", "Justin", "Noah", "Christian", "Fred", "Yohan", "Tyson", "Asafa", "Maurice", "Trayvon", 
               "Andre", "Michael", "Wayde", "Akani", "Zharnel", "Ronnie", "Jimmy", "Ramil", "Francis", "Ferdinand"]

LAST_NAMES = ["Bolt", "Gatlin", "Lyles", "Coleman", "Kerley", "Blake", "Gay", "Powell", "Greene", "Bromell", 
              "De Grasse", "Johnson", "Van Niekerk", "Simbine", "Hughes", "Baker", "Vicaut", "Guliyev", "Obikwelu", "Omanyala"]

COUNTRIES = ["JAM", "USA", "CAN", "GBR", "RSA", "FRA", "ITA", "JPN", "CHN", "BRA", "NED", "GER", "KEN", "NGR", "AUS", "ESP", "POR", "CUB"]

COMPETITIONS = ["World Championships", "Olympic Games", "Diamond League", "Continental Tour", "National Championships", 
                "European Championships", "African Championships", "Commonwealth Games", "World Indoor Championships"]

CITIES = ["Paris", "London", "Tokyo", "Berlin", "Rome", "Eugene", "Doha", "Monaco", "Zurich", "Oslo", "Brussels"]

def generate_athlete():
    """Generate a random athlete profile."""
    height = random.randint(170, 195)
    weight = round(height * 0.4 + random.uniform(-5, 10), 1)
    
    return {
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "country": random.choice(COUNTRIES),
        "height_cm": height,
        "weight_kg": weight,
        "age": random.randint(20, 35),
        "personal_best_100m": round(9.5 + random.uniform(0, 1.5), 2),
        "stride_length_m": round(2.0 + random.uniform(0, 0.5), 2),
        "reaction_time_s": round(0.12 + random.uniform(0, 0.08), 3)
    }

def generate_race_result(athletes, race_type="100m"):
    """Generate a random race result."""
    # Select random participants
    participants = random.sample(athletes, random.randint(6, 8))
    race_date = datetime.now() - timedelta(days=random.randint(0, 1000))
    
    # Generate base time based on race type
    if race_type == "100m":
        base_time = 9.7
        time_spread = 1.5
    elif race_type == "200m":
        base_time = 19.5
        time_spread = 2.5
    else:  # 400m
        base_time = 43.0
        time_spread = 4.0
    
    # Generate results
    results = []
    for i, athlete in enumerate(participants):
        # Adjust times to ensure they increase with position
        time = round(base_time + (i * time_spread / len(participants)) + random.uniform(0, 0.3), 2)
        
        results.append({
            "pos": i + 1,
            "athlete": athlete["name"],
            "country": athlete["country"],
            "mark": str(time),
            "time": str(time),
            "lane": i + 1,
            "reaction_time": round(0.12 + random.uniform(0, 0.08), 3),
            "wind": round(random.uniform(-2.0, 2.0), 1)
        })
    
    race = {
        "event": race_type,
        "date": race_date.strftime("%Y-%m-%d"),
        "location": f"{random.choice(CITIES)}, {random.choice(COUNTRIES)}",
        "competition": random.choice(COMPETITIONS),
        "round": random.choice(["Heat", "Semi-final", "Final"]),
        "results": results
    }
    
    return race

def generate_biomechanics_data(athletes):
    """Generate sprint biomechanics data."""
    biomechanics = []
    
    for athlete in athletes:
        # Generate random biomechanical data
        data = {
            "athlete": athlete["name"],
            "country": athlete["country"],
            "stride_length_m": athlete["stride_length_m"],
            "stride_frequency_hz": round(4.0 + random.uniform(-0.5, 0.5), 2),
            "ground_contact_time_ms": round(80 + random.uniform(-10, 20), 1),
            "flight_time_ms": round(120 + random.uniform(-20, 30), 1),
            "vertical_oscillation_cm": round(5 + random.uniform(-1, 3), 1),
            "leg_stiffness_kn_m": round(12 + random.uniform(-2, 4), 1),
            "hip_extension_angle_deg": round(170 + random.uniform(-10, 10), 1),
            "knee_flexion_angle_deg": round(130 + random.uniform(-15, 15), 1),
            "ankle_angle_deg": round(70 + random.uniform(-10, 10), 1),
            "peak_force_n": round(athlete["weight_kg"] * 22 + random.uniform(-200, 300), 0),
            "horizontal_velocity_ms": round(11 + random.uniform(-1, 1), 2),
            "power_output_w": round(2500 + random.uniform(-300, 500), 0)
        }
        biomechanics.append(data)
    
    return biomechanics

def create_csv_files(races, output_dir):
    """Create CSV files with race results."""
    races_dir = os.path.join(output_dir, "races")
    os.makedirs(races_dir, exist_ok=True)
    
    for i, race in enumerate(races):
        # Create a pandas DataFrame from the race results
        df = pd.DataFrame(race["results"])
        
        # Generate filename
        filename = f"{race['event']}_{race['competition'].replace(' ', '_')}_{i+1}.csv"
        filepath = os.path.join(races_dir, filename)
        
        # Save as CSV
        df.to_csv(filepath, index=False)
        print(f"Created CSV file: {filepath}")

def create_json_files(athletes, races, biomechanics, output_dir):
    """Create JSON files with various sprint data."""
    # Create directories
    athletes_dir = os.path.join(output_dir, "athletes")
    analysis_dir = os.path.join(output_dir, "analysis")
    races_dir = os.path.join(output_dir, "races_json")
    
    os.makedirs(athletes_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(races_dir, exist_ok=True)
    
    # Save athletes data
    with open(os.path.join(athletes_dir, "athletes.json"), "w") as f:
        json.dump(athletes, f, indent=2)
    
    # Save races data
    for i, race in enumerate(races):
        filename = f"{race['event']}_{race['competition'].replace(' ', '_')}_{i+1}.json"
        with open(os.path.join(races_dir, filename), "w") as f:
            json.dump(race, f, indent=2)
    
    # Save all races in one file
    with open(os.path.join(races_dir, "all_races.json"), "w") as f:
        json.dump(races, f, indent=2)
    
    # Save biomechanics data
    with open(os.path.join(analysis_dir, "biomechanics.json"), "w") as f:
        json.dump(biomechanics, f, indent=2)
    
    # Create race analysis data
    race_analysis = {
        "title": "Sprint Performance Analysis",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "analyst": "Domain-LLM Sample Generator",
        "summary": "Analysis of sprint race performances",
        "metrics": [
            {"name": "Reaction Time", "unit": "seconds", "importance": "high"},
            {"name": "Acceleration", "unit": "m/s²", "importance": "high"},
            {"name": "Top Speed", "unit": "m/s", "importance": "high"},
            {"name": "Speed Endurance", "unit": "index", "importance": "medium"},
            {"name": "Stride Length", "unit": "meters", "importance": "medium"},
            {"name": "Stride Frequency", "unit": "Hz", "importance": "medium"}
        ],
        "data": [
            {
                "athlete": athlete["name"],
                "country": athlete["country"],
                "reaction_time": round(0.12 + random.uniform(0, 0.08), 3),
                "acceleration": round(6 + random.uniform(-1, 1), 2),
                "top_speed": round(11.5 + random.uniform(-1, 1), 2),
                "speed_endurance": round(0.7 + random.uniform(-0.2, 0.2), 2),
                "stride_length": athlete["stride_length_m"],
                "stride_frequency": round(4.0 + random.uniform(-0.5, 0.5), 2)
            }
            for athlete in random.sample(athletes, 10)
        ]
    }
    
    with open(os.path.join(analysis_dir, "race_analysis.json"), "w") as f:
        json.dump(race_analysis, f, indent=2)
    
    print(f"Created JSON files in {output_dir}")

def main():
    """Generate sample sprint data files."""
    print("Generating sample sprint data...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate data
    athletes = [generate_athlete() for _ in range(NUM_ATHLETES)]
    
    races_100m = [generate_race_result(athletes, "100m") for _ in range(NUM_RACES // 3)]
    races_200m = [generate_race_result(athletes, "200m") for _ in range(NUM_RACES // 3)]
    races_400m = [generate_race_result(athletes, "400m") for _ in range(NUM_RACES // 3)]
    all_races = races_100m + races_200m + races_400m
    
    biomechanics = generate_biomechanics_data(athletes)
    
    # Create files
    create_csv_files(all_races, OUTPUT_DIR)
    create_json_files(athletes, all_races, biomechanics, OUTPUT_DIR)
    
    print(f"Sample data generation complete. Data saved to {OUTPUT_DIR}/ directory")
    print("You can now use this data with: domain-llm process --data-dir content --output-dir data/processed")

if __name__ == "__main__":
    main() 