"""
Updated Sprint Events Scraper

Uses Selenium to navigate the World Athletics website, apply filters for men's
100m, 200m, and 400m events, and extract results for LLM training.
"""

import os
import time
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
import re
import sys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup

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

class SprintResultsScraper:
    """Scraper for men's 100m, 200m, and 400m sprint results from World Athletics."""
    
    def __init__(self, headless=True, output_dir="data/sprint_data"):
        """
        Initialize the scraper.
        
        Args:
            headless: Whether to run Chrome in headless mode
            output_dir: Directory to save scraped data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")  # Updated headless mode syntax
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
        
        try:
            # Initialize webdriver using locally installed chromedriver
            service = Service('/opt/homebrew/bin/chromedriver')  # Path to Homebrew-installed chromedriver
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
            self.driver.implicitly_wait(20)  # Increased implicit wait time
            logger.info("WebDriver initialized")
        except WebDriverException as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            sys.exit(1)
    
    def __del__(self):
        """Clean up resources when the scraper is destroyed."""
        if hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def navigate_to_calendar(self):
        """Navigate to the calendar results page."""
        url = "https://worldathletics.org/competition/calendar-results?isSearchReset=true"
        logger.info(f"Navigating to: {url}")
        
        try:
            self.driver.get(url)
            
            # Wait for the page to load
            try:
                # Increased timeout to 40 seconds
                WebDriverWait(self.driver, 40).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.CompetitionsContainer"))
                )
                logger.info("Calendar page loaded successfully")
                return True
            except TimeoutException:
                logger.error("Timed out waiting for calendar page to load")
                # Take a screenshot for debugging
                self.driver.save_screenshot(f"{self.output_dir}/error_calendar_page.png")
                logger.info(f"Screenshot saved to {self.output_dir}/error_calendar_page.png")
                return False
        except WebDriverException as e:
            logger.error(f"Error navigating to calendar page: {str(e)}")
            return False
    
    def apply_filters(self, event_type, gender="men"):
        """
        Apply filters for specific events and gender.
        
        Args:
            event_type: One of: '100m', '200m', '400m'
            gender: 'men' or 'women'
            
        Returns:
            True if filters were applied successfully, False otherwise
        """
        logger.info(f"Applying filters: {gender} {event_type}")
        
        try:
            # Open filters
            filter_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Filter')]"))
            )
            filter_button.click()
            logger.info("Opened filter menu")
            
            # Wait for filter menu to appear
            WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, "//div[contains(@class, 'FilterMenuPopper')]"))
            )
            
            # Select gender
            gender_selector = f"//div[contains(text(), 'Gender')]//following-sibling::div//span[contains(text(), '{gender.title()}')]"
            gender_element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, gender_selector))
            )
            gender_element.click()
            logger.info(f"Selected gender: {gender}")
            
            # Select event category (Sprint)
            event_category = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Event Category')]//following-sibling::div//span[contains(text(), 'Sprint')]"))
            )
            event_category.click()
            logger.info("Selected event category: Sprint")
            
            # Select specific event
            event_selector = f"//div[contains(text(), 'Event')]//following-sibling::div//span[contains(text(), '{event_type}')]"
            event_element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, event_selector))
            )
            event_element.click()
            logger.info(f"Selected event: {event_type}")
            
            # Apply filters
            apply_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Apply')]"))
            )
            apply_button.click()
            logger.info("Applied filters")
            
            # Wait for results to load
            time.sleep(5)
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return False
    
    def extract_competition_links(self):
        """
        Extract links to competitions from the filtered results page.
        
        Returns:
            List of dictionaries with competition details and links
        """
        logger.info("Extracting competition links")
        
        competitions = []
        
        try:
            # Wait for competition cards to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.CompetitionCard"))
            )
            
            # Find all competition cards
            competition_cards = self.driver.find_elements(By.CSS_SELECTOR, "div.CompetitionCard")
            logger.info(f"Found {len(competition_cards)} competition cards")
            
            for card in competition_cards:
                try:
                    # Extract competition details
                    title_elem = card.find_element(By.CSS_SELECTOR, "h3")
                    title = title_elem.text if title_elem else "Unknown"
                    
                    location_elem = card.find_element(By.CSS_SELECTOR, "div.CompetitionCard_venueCountry__qCh1l")
                    location = location_elem.text if location_elem else "Unknown"
                    
                    date_elem = card.find_element(By.CSS_SELECTOR, "div.CompetitionCard_dates__KsIH4")
                    date = date_elem.text if date_elem else "Unknown"
                    
                    # Find the "Results" link
                    results_link = None
                    links = card.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        if "results" in link.get_attribute("href").lower():
                            results_link = link.get_attribute("href")
                            break
                    
                    if results_link:
                        competitions.append({
                            "Title": title,
                            "Location": location,
                            "Date": date,
                            "ResultsLink": results_link
                        })
                        logger.info(f"Added competition: {title} ({results_link})")
                    
                except Exception as e:
                    logger.warning(f"Error extracting details from competition card: {str(e)}")
            
            return competitions
            
        except Exception as e:
            logger.error(f"Error extracting competition links: {str(e)}")
            return []
    
    def scrape_results_page(self, url, event_type):
        """
        Scrape results from a specific competition results page.
        
        Args:
            url: URL of the results page
            event_type: Type of event (100m, 200m, 400m)
            
        Returns:
            DataFrame with results or None if an error occurred
        """
        logger.info(f"Scraping results from: {url}")
        
        try:
            self.driver.get(url)
            
            # Wait for results to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.ResultsTable, table.ResultsTable"))
            )
            
            # Find the event we're looking for
            event_headers = self.driver.find_elements(By.XPATH, f"//h2[contains(text(), '{event_type}')]")
            
            results_data = []
            
            for header in event_headers:
                event_name = header.text
                
                # Check if this is the correct men's event
                if not (event_type in event_name.lower() and "men" in event_name.lower()):
                    continue
                
                logger.info(f"Found event: {event_name}")
                
                # Find the results table for this event
                # We need to navigate to the parent container and then find the table
                event_section = header.find_element(By.XPATH, "./ancestor::div[contains(@class, 'ResultsEventContainer')]")
                
                # Check for table or modern div-based results
                results_tables = event_section.find_elements(By.CSS_SELECTOR, "table.ResultsTable")
                
                if results_tables:
                    # Traditional table format
                    for table in results_tables:
                        # Get table rows
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        
                        # Skip header row
                        for row in rows[1:]:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) >= 4:
                                position = cells[0].text.strip()
                                athlete = cells[1].text.strip()
                                country = cells[2].text.strip()
                                time = cells[3].text.strip()
                                
                                results_data.append({
                                    "Event": event_name,
                                    "Position": position,
                                    "Athlete": athlete,
                                    "Country": country,
                                    "Time": time,
                                    "Competition": self.driver.title,
                                    "URL": url
                                })
                else:
                    # Modern div-based format
                    result_rows = event_section.find_elements(By.CSS_SELECTOR, "div.ResultsTable_row__OXJe2, div[class*='ResultsTable_row']")
                    
                    for row in result_rows:
                        # Skip header row
                        if "header" in row.get_attribute("class").lower():
                            continue
                        
                        # Extract data from cells
                        cells = row.find_elements(By.CSS_SELECTOR, "div.ResultsTable_cell__qhbV9, div[class*='ResultsTable_cell']")
                        
                        if len(cells) >= 4:
                            position = cells[0].text.strip()
                            athlete = cells[1].text.strip()
                            country = cells[2].text.strip()
                            time = cells[3].text.strip()
                            
                            results_data.append({
                                "Event": event_name,
                                "Position": position,
                                "Athlete": athlete,
                                "Country": country,
                                "Time": time,
                                "Competition": self.driver.title,
                                "URL": url
                            })
            
            if results_data:
                return pd.DataFrame(results_data)
            else:
                logger.warning(f"No results found for {event_type} at {url}")
                return None
            
        except Exception as e:
            logger.error(f"Error scraping results page {url}: {str(e)}")
            return None
    
    def scrape_sprint_events(self):
        """
        Main method to scrape men's 100m, 200m, and 400m results.
        
        Returns:
            DataFrame with all combined sprint results
        """
        all_results = []
        event_types = ["100m", "200m", "400m"]
        
        try:
            # Navigate to calendar page
            if not self.navigate_to_calendar():
                return None
            
            # Process each event type
            for event_type in event_types:
                logger.info(f"Processing event type: {event_type}")
                
                # Navigate to calendar and apply filters
                self.navigate_to_calendar()
                if not self.apply_filters(event_type, gender="men"):
                    continue
                
                # Extract competition links
                competitions = self.extract_competition_links()
                if not competitions:
                    logger.warning(f"No competitions found for {event_type}")
                    continue
                
                # Save competitions to CSV
                comps_df = pd.DataFrame(competitions)
                comps_df.to_csv(f"{self.output_dir}/competitions_{event_type}.csv", index=False)
                
                # Scrape results from each competition
                for i, comp in enumerate(tqdm(competitions, desc=f"Scraping {event_type} competitions")):
                    results_df = self.scrape_results_page(comp["ResultsLink"], event_type)
                    
                    if results_df is not None and not results_df.empty:
                        # Save individual results
                        results_df.to_csv(f"{self.output_dir}/{event_type}_comp_{i}.csv", index=False)
                        all_results.append(results_df)
                        logger.info(f"Extracted {len(results_df)} results from {comp['Title']}")
                    
                    # Be respectful to the server
                    time.sleep(2)
            
            # Combine all results
            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.to_csv(f"{self.output_dir}/all_sprint_results.csv", index=False)
                logger.info(f"Successfully extracted {len(final_df)} total sprint results")
                return final_df
            else:
                logger.warning("No sprint results extracted")
                return None
                
        except Exception as e:
            logger.error(f"Error in scrape_sprint_events: {str(e)}")
            return None
        finally:
            # Clean up
            self.driver.quit()
    
    def convert_to_text_corpus(self, df, output_file="data/processed/sprint_corpus.txt"):
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
--- START OF DOCUMENT: {row.get('Event', 'Unknown Event')} {row.get('Competition', 'Unknown Competition')} ---

Event: {row.get('Event', 'Unknown Event')}
Competition: {row.get('Competition', 'Unknown Competition')}
Athlete: {row.get('Athlete', 'Unknown Athlete')}
Country: {row.get('Country', 'Unknown Country')}
Position: {row.get('Position', 'Unknown Position')}
Time: {row.get('Time', 'Unknown Time')}

--- END OF DOCUMENT: {row.get('Event', 'Unknown Event')} {row.get('Competition', 'Unknown Competition')} ---
"""
            text_passages.append(passage)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_passages))
        
        logger.info(f"Created text corpus with {len(text_passages)} documents at {output_file}")
        return output_file


if __name__ == "__main__":
    # Run the scraper
    print("Starting sprint results scraper...")
    scraper = SprintResultsScraper(headless=False)  # Set headless=True for production
    
    try:
        # Run the scraping process
        sprint_data = scraper.scrape_sprint_events()
        
        if sprint_data is not None:
            # Convert to text corpus for LLM training
            print("Converting results to text corpus...")
            corpus_file = scraper.convert_to_text_corpus(sprint_data)
            print(f"Text corpus created at: {corpus_file}")
            print("You can now use this corpus for LLM training with the command:")
            print("python -m src.main train train-model --data-dir data/processed --model-dir models")
        else:
            print("No sprint data was collected. Please check the logs for details.")
            
    except Exception as e:
        print(f"Error running scraper: {str(e)}")
    finally:
        # Clean up
        if hasattr(scraper, 'driver'):
            scraper.driver.quit() 