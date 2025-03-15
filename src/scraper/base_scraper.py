"""
Base Scraper Module

Defines the base scraper class that all specific scrapers should inherit from.
"""

import os
import time
import logging
import pandas as pd
import requests
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from .supabase_handler import SupabaseHandler
from .config import ScraperConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)


class BaseScraper(ABC):
    """
    Base scraper class that defines the interface and common functionality
    for all scrapers in the system.
    """
    
    def __init__(self, config: ScraperConfig, supabase_handler: Optional[SupabaseHandler] = None):
        """
        Initialize the base scraper with configuration.
        
        Args:
            config: ScraperConfig object with settings
            supabase_handler: Optional SupabaseHandler for database storage
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supabase = supabase_handler
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent
        })
        
        # Set up output directories
        self.data_dir = config.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def scrape_list_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape a list/index page to get URLs for detailed pages.
        
        Args:
            url: URL of the list page
            
        Returns:
            List of dictionaries with extracted data and detail URLs
        """
        pass
    
    @abstractmethod
    def scrape_detail_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape a detail page for specific information.
        
        Args:
            url: URL of the detail page
            
        Returns:
            Dictionary with extracted data
        """
        pass
    
    def request_with_retry(self, url: str, max_retries: int = 3, 
                           delay: int = 2) -> Optional[requests.Response]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            url: URL to request
            max_retries: Maximum number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All retries failed for URL: {url}")
                    return None
    
    def throttle(self) -> None:
        """
        Apply rate limiting to be respectful of the server.
        """
        time.sleep(self.config.request_delay)
    
    def save_to_csv(self, data: Union[pd.DataFrame, List[Dict]], filename: str) -> str:
        """
        Save scraped data to a CSV file.
        
        Args:
            data: DataFrame or list of dictionaries to save
            filename: Base filename (without extension)
            
        Returns:
            Path to the saved file
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_dir, f"{filename}_{timestamp}.csv")
        
        data.to_csv(filepath, index=False)
        self.logger.info(f"Saved data to {filepath}")
        
        return filepath
    
    def save_to_supabase(self, data: Union[pd.DataFrame, List[Dict]], table_name: str) -> bool:
        """
        Save scraped data to Supabase.
        
        Args:
            data: DataFrame or list of dictionaries to save
            table_name: Name of the table to insert into
            
        Returns:
            True if successful, False otherwise
        """
        if self.supabase is None:
            self.logger.warning("No Supabase handler provided, skipping database save")
            return False
        
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        
        try:
            result = self.supabase.insert_data(table_name, data)
            self.logger.info(f"Saved {len(data)} records to Supabase table: {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save data to Supabase: {str(e)}")
            return False
    
    def run(self, list_urls: List[str], save_csv: bool = True, 
            save_to_db: bool = True, table_name: str = None) -> pd.DataFrame:
        """
        Run the full scraping process.
        
        Args:
            list_urls: List of URLs to scrape list pages from
            save_csv: Whether to save results to CSV
            save_to_db: Whether to save results to Supabase
            table_name: Supabase table name (required if save_to_db is True)
            
        Returns:
            DataFrame with all scraped data
        """
        if save_to_db and (table_name is None or self.supabase is None):
            raise ValueError("Table name and Supabase handler must be provided when save_to_db is True")
        
        self.logger.info(f"Starting scraping process with {len(list_urls)} list URLs")
        all_detail_info = []
        
        # Process each list page
        for list_url in list_urls:
            self.logger.info(f"Processing list URL: {list_url}")
            detail_infos = self.scrape_list_page(list_url)
            self.throttle()
            
            # Process each detail page from the list
            for detail_info in detail_infos:
                if 'detail_url' in detail_info and detail_info['detail_url']:
                    self.logger.info(f"Processing detail URL: {detail_info['detail_url']}")
                    detail_data = self.scrape_detail_page(detail_info['detail_url'])
                    self.throttle()
                    
                    # Merge list and detail data
                    merged_data = {**detail_info, **detail_data}
                    all_detail_info.append(merged_data)
                else:
                    all_detail_info.append(detail_info)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_detail_info)
        
        # Save results if requested
        if save_csv:
            self.save_to_csv(results_df, self.__class__.__name__)
        
        if save_to_db:
            self.save_to_supabase(results_df, table_name)
        
        self.logger.info(f"Scraping complete. {len(results_df)} records collected.")
        return results_df 