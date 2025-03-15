"""
Data Processing Module.

This module provides functionality for processing and extracting text from various
data sources, particularly academic papers in PDF format.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import concurrent.futures
import re

import PyPDF2
import pdfplumber
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Class for processing PDF files and extracting text."""
    
    def __init__(self, papers_dir: str = "papers"):
        """
        Initialize the PDF processor.
        
        Args:
            papers_dir: Directory containing PDF papers
        """
        self.papers_dir = Path(papers_dir)
        if not self.papers_dir.exists():
            raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
        
        self.pdf_files = list(self.papers_dir.glob("**/*.pdf"))
        logger.info(f"Found {len(self.pdf_files)} PDF files in {papers_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file using multiple extraction methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return ""
        
        # Try PyPDF2 first
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                if len(text.strip()) > 100:  # If we got substantial text
                    return self._clean_text(text)
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {e}")
        
        # Fall back to pdfplumber if PyPDF2 fails or extracts little text
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or "" + "\n\n"
                return self._clean_text(text)
        except Exception as e:
            logger.error(f"All PDF extraction methods failed for {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove headers and footers (simplistic approach)
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip page numbers, headers, etc.
            if re.match(r'^\s*\d+\s*$', line):  # Just page numbers
                continue
            if len(line.strip()) < 3:  # Very short lines
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def process_all_papers(self, max_workers: int = 4) -> Dict[str, str]:
        """
        Process all PDF papers in parallel.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        results = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.extract_text_from_pdf, pdf_file): pdf_file
                for pdf_file in self.pdf_files
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_pdf), 
                              total=len(self.pdf_files), 
                              desc="Processing PDFs"):
                pdf_file = future_to_pdf[future]
                try:
                    text = future.result()
                    if text:
                        results[pdf_file.name] = text
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
        
        logger.info(f"Successfully extracted text from {len(results)} out of {len(self.pdf_files)} PDFs")
        return results
    
    def save_extracted_texts(self, output_dir: str = "data/processed") -> None:
        """
        Extract text from all PDFs and save to files.
        
        Args:
            output_dir: Directory to save extracted texts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted_texts = self.process_all_papers()
        
        for filename, text in extracted_texts.items():
            output_file = output_path / f"{Path(filename).stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Create a combined file with all texts
        combined_file = output_path / "all_papers_combined.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for filename, text in extracted_texts.items():
                f.write(f"\n\n--- START OF DOCUMENT: {filename} ---\n\n")
                f.write(text)
                f.write(f"\n\n--- END OF DOCUMENT: {filename} ---\n\n")
        
        logger.info(f"Saved extracted texts to {output_dir}")
        
        # Create a metadata file
        metadata = pd.DataFrame([
            {"filename": k, "char_count": len(v), "word_count": len(v.split())}
            for k, v in extracted_texts.items()
        ])
        metadata.to_csv(output_path / "metadata.csv", index=False)


if __name__ == "__main__":
    processor = PDFProcessor()
    processor.save_extracted_texts() 