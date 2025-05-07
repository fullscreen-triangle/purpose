"""
Supabase Handler Module

Defines the SupabaseHandler class for interacting with Supabase.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from supabase import create_client, Client


class SupabaseHandler:
    """
    Handler for interacting with Supabase database.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase client.
        
        Args:
            supabase_url: URL of the Supabase instance
            supabase_key: API key for the Supabase instance
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be provided")
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            self.logger.info("Successfully connected to Supabase")
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    def insert_data(self, table_name: str, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert data into a Supabase table.
        
        Args:
            table_name: Name of the table to insert into
            data: List of dictionaries or single dictionary with data to insert
            
        Returns:
            Response from Supabase
        """
        if not data:
            self.logger.warning("No data to insert")
            return {'count': 0}
        
        # Convert single item to list
        if isinstance(data, dict):
            data = [data]
        
        try:
            response = self.client.table(table_name).insert(data).execute()
            self.logger.info(f"Inserted {len(data)} records into {table_name}")
            return response.data
        except Exception as e:
            self.logger.error(f"Failed to insert data into {table_name}: {str(e)}")
            raise
    
    def upsert_data(self, table_name: str, data: Union[List[Dict[str, Any]], Dict[str, Any]], 
                   on_conflict: str) -> Dict[str, Any]:
        """
        Upsert data into a Supabase table.
        
        Args:
            table_name: Name of the table to upsert into
            data: List of dictionaries or single dictionary with data to upsert
            on_conflict: Column name to determine conflicts
            
        Returns:
            Response from Supabase
        """
        if not data:
            self.logger.warning("No data to upsert")
            return {'count': 0}
        
        # Convert single item to list
        if isinstance(data, dict):
            data = [data]
        
        try:
            response = self.client.table(table_name).upsert(data, on_conflict=on_conflict).execute()
            self.logger.info(f"Upserted {len(data)} records into {table_name}")
            return response.data
        except Exception as e:
            self.logger.error(f"Failed to upsert data into {table_name}: {str(e)}")
            raise
    
    def fetch_data(self, table_name: str, columns: str = "*", 
                  filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from a Supabase table.
        
        Args:
            table_name: Name of the table to fetch from
            columns: Columns to select (default: all)
            filters: Dictionary of filters to apply
            
        Returns:
            DataFrame with fetched data
        """
        query = self.client.table(table_name).select(columns)
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)
        
        try:
            response = query.execute()
            self.logger.info(f"Fetched {len(response.data)} records from {table_name}")
            return pd.DataFrame(response.data)
        except Exception as e:
            self.logger.error(f"Failed to fetch data from {table_name}: {str(e)}")
            raise
    
    def check_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        try:
            # Try to fetch schema information
            response = self.client.table(table_name).select("*").limit(1).execute()
            return True
        except Exception:
            return False
    
    def delete_data(self, table_name: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete data from a Supabase table.
        
        Args:
            table_name: Name of the table to delete from
            filters: Dictionary of filters to apply
            
        Returns:
            Response from Supabase
        """
        if not filters:
            self.logger.warning("No filters provided for delete operation. Refusing to delete all records.")
            return {'count': 0}
        
        query = self.client.table(table_name).delete()
        
        # Apply filters
        for column, value in filters.items():
            query = query.eq(column, value)
        
        try:
            response = query.execute()
            self.logger.info(f"Deleted records from {table_name} with filters: {filters}")
            return response.data
        except Exception as e:
            self.logger.error(f"Failed to delete data from {table_name}: {str(e)}")
            raise 