"""
Continuous Learning Module.

This module provides functionality for continuously updating language models
as new sprint running papers and data become available.
"""

import os
import logging
import json
import datetime
from pathlib import Path
import shutil
from typing import List, Dict, Optional, Union, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM
)
from peft import PeftModel, PeftConfig

from core.data_processing import PDFProcessor
from core.model_training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousLearner:
    """Class for continuously updating language models with new data."""
    
    def __init__(self,
                papers_dir: str = "papers",
                processed_data_dir: str = "data/processed",
                models_dir: str = "models",
                tracking_file: str = "data/continuous_learning_tracker.json"):
        """
        Initialize the continuous learner.
        
        Args:
            papers_dir: Directory containing PDF papers
            processed_data_dir: Directory for processed text data
            models_dir: Directory containing saved models
            tracking_file: JSON file to track processed papers and model versions
        """
        self.papers_dir = Path(papers_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.tracking_file = Path(tracking_file)
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize tracking data
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict[str, Any]:
        """
        Load tracking data from JSON file or initialize if not exists.
        
        Returns:
            Dictionary of tracking data
        """
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty tracking data
            tracking_data = {
                "processed_papers": [],
                "model_versions": [],
                "last_update": None
            }
            # Save initial tracking data
            with open(self.tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            return tracking_data
    
    def _save_tracking_data(self):
        """Save tracking data to JSON file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def _get_latest_model(self) -> Optional[str]:
        """
        Get the path to the latest model.
        
        Returns:
            Path to the latest model directory or None if no models exist
        """
        if not self.tracking_data["model_versions"]:
            # Find most recent model directory by creation time
            model_dirs = list(self.models_dir.glob("sprint-llm-*"))
            if not model_dirs:
                return None
            
            latest_model = max(model_dirs, key=lambda x: x.stat().st_mtime)
            return str(latest_model)
        
        # Return the most recent model from tracking data
        return self.tracking_data["model_versions"][-1]["path"]
    
    def check_for_new_papers(self) -> List[Path]:
        """
        Check for new papers that haven't been processed yet.
        
        Returns:
            List of paths to new papers
        """
        # Get all PDF files in the papers directory
        all_papers = list(self.papers_dir.glob("**/*.pdf"))
        
        # Get filenames of already processed papers
        processed_filenames = set(
            paper["filename"] for paper in self.tracking_data["processed_papers"]
        )
        
        # Filter for new papers
        new_papers = [
            paper for paper in all_papers
            if paper.name not in processed_filenames
        ]
        
        logger.info(f"Found {len(new_papers)} new papers to process")
        return new_papers
    
    def process_new_papers(self) -> bool:
        """
        Process new papers and update tracking data.
        
        Returns:
            True if new papers were processed, False otherwise
        """
        # Check for new papers
        new_papers = self.check_for_new_papers()
        if not new_papers:
            logger.info("No new papers to process")
            return False
        
        # Create a processor just for the new papers
        processor = PDFProcessor(papers_dir=str(self.papers_dir))
        
        # Process new papers and save to processed data directory
        for paper_path in new_papers:
            logger.info(f"Processing paper: {paper_path.name}")
            
            # Extract text
            text = processor.extract_text_from_pdf(paper_path)
            
            if not text:
                logger.warning(f"Failed to extract text from {paper_path.name}")
                continue
            
            # Save processed text
            output_file = self.processed_data_dir / f"{paper_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Update tracking data
            self.tracking_data["processed_papers"].append({
                "filename": paper_path.name,
                "processed_at": datetime.datetime.now().isoformat(),
                "output_path": str(output_file),
                "char_count": len(text),
                "word_count": len(text.split())
            })
        
        # Save updated tracking data
        self._save_tracking_data()
        
        # Update the combined file
        self._update_combined_file()
        
        return True
    
    def _update_combined_file(self):
        """Update the combined file with all processed papers."""
        combined_file = self.processed_data_dir / "all_papers_combined.txt"
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            for paper_data in self.tracking_data["processed_papers"]:
                paper_file = Path(paper_data["output_path"])
                
                if paper_file.exists():
                    with open(paper_file, 'r', encoding='utf-8') as paper_f:
                        text = paper_f.read()
                    
                    f.write(f"\n\n--- START OF DOCUMENT: {paper_data['filename']} ---\n\n")
                    f.write(text)
                    f.write(f"\n\n--- END OF DOCUMENT: {paper_data['filename']} ---\n\n")
    
    def update_model(self, 
                    force_update: bool = False,
                    batch_size: int = 8,
                    learning_rate: float = 5e-5,
                    num_epochs: int = 3) -> Optional[str]:
        """
        Update the model if new papers are available.
        
        Args:
            force_update: Whether to force update even if no new papers
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            
        Returns:
            Path to the updated model or None if no update was performed
        """
        # Check and process new papers
        new_papers_processed = self.process_new_papers()
        
        if not new_papers_processed and not force_update:
            logger.info("No new papers processed and force_update is False. Skipping model update.")
            return None
        
        # Get latest model
        latest_model_path = self._get_latest_model()
        
        # Initialize trainer with appropriate model
        if latest_model_path:
            logger.info(f"Continuing training from latest model: {latest_model_path}")
            trainer = ModelTrainer(
                processed_data_dir=str(self.processed_data_dir),
                model_dir=str(self.models_dir),
                model_name=latest_model_path
            )
        else:
            logger.info("No existing model found. Training from pretrained model.")
            trainer = ModelTrainer(
                processed_data_dir=str(self.processed_data_dir),
                model_dir=str(self.models_dir)
            )
        
        # Train model
        _, metrics = trainer.train(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        
        # Get path to newly trained model
        model_dirs = list(self.models_dir.glob("sprint-llm-*"))
        if not model_dirs:
            logger.error("No model found after training")
            return None
        
        new_model_path = str(max(model_dirs, key=lambda x: x.stat().st_mtime))
        
        # Update tracking data
        self.tracking_data["model_versions"].append({
            "path": new_model_path,
            "created_at": datetime.datetime.now().isoformat(),
            "num_papers": len(self.tracking_data["processed_papers"]),
            "metrics": metrics
        })
        
        self.tracking_data["last_update"] = datetime.datetime.now().isoformat()
        self._save_tracking_data()
        
        logger.info(f"Model updated successfully: {new_model_path}")
        return new_model_path
    
    def run_continuous_learning_cycle(self):
        """Run a complete continuous learning cycle."""
        logger.info("Starting continuous learning cycle")
        
        # Process new papers
        new_papers = self.process_new_papers()
        
        if new_papers:
            # Update model with new papers
            updated_model_path = self.update_model()
            
            if updated_model_path:
                logger.info(f"Continuous learning cycle completed. New model: {updated_model_path}")
            else:
                logger.warning("Continuous learning cycle completed but model update failed")
        else:
            logger.info("No new papers to process. Continuous learning cycle complete with no changes.")


if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.run_continuous_learning_cycle() 