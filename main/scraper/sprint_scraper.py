"""
Sprint Events Scraper

This script uses the existing World Athletics scraper to specifically
extract data for men's 100m, 200m, and 400m events.
"""

import os
import pandas as pd
import re
from pathlib import Path
import logging
from tqdm import tqdm

from scraper.worldathletics_scraper import first_page_scrape, second_page, check_url, scrape_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sprint_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sprint-scraper")

def scrape_sprint_events(output_dir="data/sprint_data", verify_ssl=True):
    """
    Scrape specifically men's 100m, 200m, and 400m events from World Athletics.
    
    Args:
        output_dir: Directory to save the results
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        DataFrame containing the combined sprint data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Get all championships
    logger.info("Getting list of championships")
    df_championships = first_page_scrape(verify_ssl=verify_ssl)
    df_championships.to_csv(f"{output_dir}/championships.csv", index=False)
    logger.info(f"Found {len(df_championships)} championships")
    
    # Step 2: Get event lists for each championship
    sprint_events_pattern = re.compile(r'.*men.*(?:100m|200m|400m).*final.*', re.IGNORECASE)
    all_sprint_events = []
    
    logger.info("Processing championships to find sprint events")
    for idx, row in tqdm(df_championships.iterrows(), total=len(df_championships), desc="Processing championships"):
        meeting = row['Meeting']
        url = row['URL']
        results_path = f"{output_dir}/results_{idx}.csv"
        
        # Get all events for this championship
        events_df = second_page(url, results_path, verify_ssl=verify_ssl)
        
        if events_df is not None and not events_df.empty:
            # Filter for men's 100m, 200m, 400m events
            sprint_events = events_df[events_df['Event'].str.match(sprint_events_pattern, na=False)]
            
            if not sprint_events.empty:
                # Add championship info to the events
                sprint_events['Championship'] = meeting
                sprint_events['Championship_URL'] = url
                all_sprint_events.append(sprint_events)
                
                logger.info(f"Found {len(sprint_events)} sprint events in {meeting}")
            else:
                logger.info(f"No sprint events found in {meeting}")
    
    # Combine all sprint events
    if all_sprint_events:
        combined_events = pd.concat(all_sprint_events, ignore_index=True)
        combined_events.to_csv(f"{output_dir}/all_sprint_events.csv", index=False)
        logger.info(f"Found a total of {len(combined_events)} sprint events across all championships")
    else:
        logger.warning("No sprint events found in any championship")
        return None
    
    # Step 3: Get detailed results for each sprint event
    all_results = []
    
    logger.info("Getting detailed results for each sprint event")
    for idx, row in tqdm(combined_events.iterrows(), total=len(combined_events), desc="Extracting sprint results"):
        event_name = row['Event']
        url = row['URL']
        championship = row['Championship']
        
        try:
            results_df = check_url(url, verify_ssl=verify_ssl)
            
            if results_df is not None and not results_df.empty:
                # Add event and championship info
                results_df['Event_Name'] = event_name
                results_df['Championship'] = championship
                
                # Save individual event results
                results_df.to_csv(f"{output_dir}/event_{idx}.csv", index=False)
                
                all_results.append(results_df)
                logger.info(f"Extracted {len(results_df)} results for {event_name} in {championship}")
            else:
                logger.warning(f"No results found for {event_name}")
        except Exception as e:
            logger.error(f"Error extracting data for {event_name}: {str(e)}")
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(f"{output_dir}/all_sprint_results.csv", index=False)
        logger.info(f"Successfully extracted {len(final_df)} sprint results across all events")
        return final_df
    else:
        logger.warning("No sprint results extracted")
        return None

def convert_to_text_corpus(df, output_file="data/processed/sprint_corpus.txt"):
    """
    Convert the sprint results DataFrame to a text corpus for LLM training.
    
    Args:
        df: DataFrame with sprint results
        output_file: Path to save the text corpus
        
    Returns:
        Path to the saved text corpus file
    """
    if df is None or df.empty:
        logger.error("No data to convert to text corpus")
        return None
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format each result as a text passage
    text_passages = []
    
    for _, row in df.iterrows():
        # Create a detailed text description of the result
        passage = f"""
--- START OF DOCUMENT: {row.get('Event_Name', 'Unknown Event')} {row.get('Edition', 'Unknown Edition')} ---

Event: {row.get('Event_Name', 'Unknown Event')}
Championship: {row.get('Championship', 'Unknown Championship')}
Edition: {row.get('Edition', 'Unknown Edition')}
Athlete: {row.get('Athlete', 'Unknown Athlete')}
Country: {row.get('Country', 'Unknown Country')}
Position: {row.get('Position', 'Unknown Position')}
Result: {row.get('Mark', 'Unknown Mark')}

--- END OF DOCUMENT: {row.get('Event_Name', 'Unknown Event')} {row.get('Edition', 'Unknown Edition')} ---
"""
        text_passages.append(passage)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(text_passages))
    
    logger.info(f"Created text corpus with {len(text_passages)} documents at {output_file}")
    return output_file

if __name__ == "__main__":
    # Run the scraper for sprint events
    print("Starting sprint events scraper...")
    sprint_data = scrape_sprint_events(verify_ssl=False)  # Set to False if SSL issues occur
    
    if sprint_data is not None:
        # Convert to text corpus for LLM training
        print("Converting results to text corpus...")
        corpus_file = convert_to_text_corpus(sprint_data)
        print(f"Text corpus created at: {corpus_file}")
        print("You can now use this corpus for LLM training with the command:")
        print("python -m src.main train train-model --data-dir data/processed --model-dir models")
    else:
        print("No sprint data was collected. Please check the logs for details.") 