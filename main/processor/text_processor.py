"""
Text Processor for Sprint Results Data

This module provides functionality to clean and prepare sprint results data for LLM training.
"""

import os
import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("text-processor")

class SprintTextProcessor:
    """Processor for cleaning and preparing sprint results data for LLM training."""
    
    def __init__(self, input_dir: str = "data/sprint_data", output_dir: str = "data/processed"):
        """
        Initialize the text processor.
        
        Args:
            input_dir: Directory containing the raw sprint data
            output_dir: Directory to save the processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_sprint_data(self, filename: str = "all_sprint_results.csv") -> pd.DataFrame:
        """
        Load sprint results data from CSV.
        
        Args:
            filename: Name of the CSV file in the input directory
            
        Returns:
            DataFrame with sprint results
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Sprint data file not found: {file_path}")
        
        logger.info(f"Loading sprint data from {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} sprint results")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the sprint results data.
        
        Args:
            df: DataFrame with sprint results
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning sprint data")
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Fill missing values
        cleaned_df = cleaned_df.fillna({
            "Event": "Unknown Event",
            "Position": "Unknown",
            "Athlete": "Unknown Athlete",
            "Country": "Unknown Country",
            "Time": "Unknown Time",
            "Competition": "Unknown Competition",
            "URL": ""
        })
        
        # Clean text fields
        for col in ["Event", "Athlete", "Country", "Competition"]:
            if col in cleaned_df.columns:
                # Remove extra whitespace
                cleaned_df[col] = cleaned_df[col].str.strip()
                # Replace multiple spaces with a single space
                cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Clean time field
        if "Time" in cleaned_df.columns:
            # Remove non-numeric characters except for dots and colons
            cleaned_df["Time"] = cleaned_df["Time"].str.replace(r'[^0-9:.]', '', regex=True)
        
        # Clean position field
        if "Position" in cleaned_df.columns:
            # Extract numeric position if possible
            cleaned_df["Position"] = cleaned_df["Position"].str.extract(r'(\d+)', expand=False)
            cleaned_df["Position"] = cleaned_df["Position"].fillna("Unknown")
        
        logger.info("Data cleaning completed")
        return cleaned_df
    
    def create_text_corpus(
        self, 
        df: pd.DataFrame, 
        output_file: str = "sprint_corpus.txt",
        include_headers: bool = True
    ) -> str:
        """
        Convert the sprint results DataFrame to a text corpus for LLM training.
        
        Args:
            df: DataFrame with sprint results
            output_file: Name of the output file
            include_headers: Whether to include document headers
            
        Returns:
            Path to the saved text corpus file
        """
        logger.info("Creating text corpus")
        
        # Format each result as a text passage
        text_passages = []
        
        for _, row in df.iterrows():
            # Create a detailed text description of the result
            if include_headers:
                passage = f"""
--- START OF DOCUMENT: {row.get('Event', 'Unknown Event')} {row.get('Competition', 'Unknown Competition')} ---

Event: {row.get('Event', 'Unknown Event')}
Competition: {row.get('Competition', 'Unknown Competition')}
Athlete: {row.get('Athlete', 'Unknown Athlete')}
Country: {row.get('Country', 'Unknown Country')}
Position: {row.get('Position', 'Unknown Position')}
Time: {row.get('Time', 'Unknown Time')}

--- END OF DOCUMENT: {row.get('Event', 'Unknown Event')} {row.get('Competition', 'Unknown Competition')} ---
"""
            else:
                passage = f"""
Event: {row.get('Event', 'Unknown Event')}
Competition: {row.get('Competition', 'Unknown Competition')}
Athlete: {row.get('Athlete', 'Unknown Athlete')}
Country: {row.get('Country', 'Unknown Country')}
Position: {row.get('Position', 'Unknown Position')}
Time: {row.get('Time', 'Unknown Time')}
"""
            text_passages.append(passage)
        
        # Write to output file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_passages))
        
        logger.info(f"Created text corpus with {len(text_passages)} documents at {output_path}")
        return str(output_path)
    
    def create_qa_pairs(self, df: pd.DataFrame, output_file: str = "sprint_qa_pairs.txt") -> str:
        """
        Create question-answer pairs from sprint results for instruction tuning.
        
        Args:
            df: DataFrame with sprint results
            output_file: Name of the output file
            
        Returns:
            Path to the saved QA pairs file
        """
        logger.info("Creating question-answer pairs")
        
        qa_pairs = []
        
        # Generate QA pairs for each result
        for _, row in df.iterrows():
            event = row.get('Event', 'Unknown Event')
            competition = row.get('Competition', 'Unknown Competition')
            athlete = row.get('Athlete', 'Unknown Athlete')
            country = row.get('Country', 'Unknown Country')
            position = row.get('Position', 'Unknown Position')
            time = row.get('Time', 'Unknown Time')
            
            # Generate various question types
            qa_pairs.extend([
                {
                    "question": f"Who won the {event} at the {competition}?",
                    "answer": f"{athlete} from {country} won the {event} at the {competition} with a time of {time}."
                },
                {
                    "question": f"What was {athlete}'s time in the {event} at {competition}?",
                    "answer": f"{athlete} ran a time of {time} in the {event} at {competition}."
                },
                {
                    "question": f"Which country does {athlete} represent?",
                    "answer": f"{athlete} represents {country}."
                },
                {
                    "question": f"What position did {athlete} finish in the {event} at {competition}?",
                    "answer": f"{athlete} finished in position {position} in the {event} at {competition}."
                }
            ])
        
        # Format QA pairs as text
        qa_text = []
        for pair in qa_pairs:
            qa_text.append(f"Question: {pair['question']}\nAnswer: {pair['answer']}")
        
        # Write to output file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(qa_text))
        
        logger.info(f"Created {len(qa_pairs)} QA pairs at {output_path}")
        return str(output_path)
    
    def process_data(self) -> Dict[str, str]:
        """
        Process the sprint results data and create training corpora.
        
        Returns:
            Dictionary with paths to the created files
        """
        try:
            # Load data
            df = self.load_sprint_data()
            
            # Clean data
            cleaned_df = self.clean_data(df)
            
            # Save cleaned data
            cleaned_csv_path = self.output_dir / "cleaned_sprint_results.csv"
            cleaned_df.to_csv(cleaned_csv_path, index=False)
            logger.info(f"Saved cleaned data to {cleaned_csv_path}")
            
            # Create text corpus
            corpus_path = self.create_text_corpus(cleaned_df)
            
            # Create QA pairs
            qa_path = self.create_qa_pairs(cleaned_df)
            
            return {
                "cleaned_data": str(cleaned_csv_path),
                "text_corpus": corpus_path,
                "qa_pairs": qa_path
            }
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

def process_sprint_data(
    input_dir: str = "data/sprint_data",
    output_dir: str = "data/processed"
) -> Dict[str, str]:
    """
    Process sprint results data for LLM training.
    
    Args:
        input_dir: Directory containing the raw sprint data
        output_dir: Directory to save the processed data
        
    Returns:
        Dictionary with paths to the created files
    """
    try:
        processor = SprintTextProcessor(input_dir=input_dir, output_dir=output_dir)
        return processor.process_data()
    except Exception as e:
        logger.error(f"Error in process_sprint_data: {str(e)}")
        raise 