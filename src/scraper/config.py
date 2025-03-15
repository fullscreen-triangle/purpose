"""
Scraper Configuration Module

Defines the configuration classes used by the scraper system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ScraperConfig:
    """
    Configuration class for scrapers.
    
    Attributes:
        data_dir: Directory where scraped data will be saved
        request_delay: Delay between requests in seconds
        request_timeout: HTTP request timeout in seconds
        user_agent: User agent string for HTTP requests
        supabase_url: URL of the Supabase instance
        supabase_key: API key for the Supabase instance
        selectors: CSS selectors for different data elements (source-specific)
    """
    
    data_dir: str = "data/scraped"
    request_delay: float = 1.5
    request_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    selectors: Dict[str, Any] = None
    
    def __post_init__(self):
        """
        Initialize default values and validate config.
        """
        # Create default selectors if none provided
        if self.selectors is None:
            self.selectors = {}
        
        # Try to load Supabase credentials from environment variables if not provided
        if self.supabase_url is None:
            self.supabase_url = os.environ.get('SUPABASE_URL')
        
        if self.supabase_key is None:
            self.supabase_key = os.environ.get('SUPABASE_KEY')
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ScraperConfig':
        """
        Create a config instance from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            ScraperConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ScraperConfig':
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            ScraperConfig instance
        """
        import json
        import yaml
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.endswith(('.yaml', '.yml')):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return cls.from_dict(config_dict)


class WorldAthleticsConfig(ScraperConfig):
    """
    Configuration specific to World Athletics scraper.
    """
    
    def __init__(self, **kwargs):
        # Default selectors for World Athletics
        default_selectors = {
            'list_page': {
                'event_rows': 'table.records-table tbody tr',
                'event_name': 'td:nth-child(1)',
                'athlete_name': 'td:nth-child(4)',
                'result': 'td:nth-child(2)',
                'date': 'td:nth-child(5)',
                'detail_link': 'td:nth-child(1) a',
            },
            'detail_page': {
                'result_rows': 'table.records-table tbody tr',
                'round': 'td:nth-child(1)',
                'position': 'td:nth-child(2)',
                'athlete': 'td:nth-child(3)',
                'country': 'td:nth-child(4)',
                'mark': 'td:nth-child(5)',
                'wind': 'td:nth-child(6)',
                'date': 'td:nth-child(8)',
            }
        }
        
        # Override with any provided selectors
        selectors = kwargs.pop('selectors', {})
        merged_selectors = {**default_selectors, **selectors}
        
        # Initialize parent class
        super().__init__(selectors=merged_selectors, **kwargs) 