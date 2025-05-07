import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time

def first_page_scrape(verify_ssl=True):
    url = "https://www.worldathletics.org/results/world-athletics-championships"
    response = requests.get(url, verify=verify_ssl)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    data = []
    for item in soup.find_all('a', class_='EventResults_link__2REwE'):
        meeting = item.find('div', class_='EventResults_meeting__2dj_O').text
        venue = item.find('div', class_='EventResults_venue__2JzYb').text
        country = item.find('div', class_='EventResults_country__3oH2O').text
        date = item.find('div', class_='EventResults_date__2_VQI').text
        url = "https://www.worldathletics.org" + item['href']
        
        data.append({
            'Meeting': meeting,
            'Venue': venue,
            'Country': country,
            'Date': date,
            'URL': url
        })
    
    return pd.DataFrame(data)

def second_page(url, path, verify_ssl=True):
    response = requests.get(url, verify=verify_ssl)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results_urls = []
    pattern = re.compile(r'.*final.*', re.IGNORECASE)
    
    for link in soup.find_all('a', href=True):
        if pattern.match(str(link.text)):
            results_urls.append({
                'URL': "https://www.worldathletics.org" + link['href'],
                'Event': link.text
            })
    
    if results_urls:
        df = pd.DataFrame(results_urls)
        df.to_csv(path, index=False)
        return df
    return None

def check_url(url, verify_ssl=True):
    response = requests.get(url, verify=verify_ssl)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    data = []
    event = soup.find('h1', class_='EventHeader_title__3Qf5E').text
    edition = soup.find('div', class_='EventHeader_meeting__1e-q9').text
    
    results = soup.find_all('tr', class_='ResultsList_row__3Z4Wd')
    
    for result in results:
        position = result.find('td', class_='ResultsList_pos__2nN8O').text
        athlete = result.find('a', class_='ResultsList_name__3mxhp').text
        country = result.find('span', class_='ResultsList_country__3KsQx').text
        mark = result.find('td', class_='ResultsList_result__1_LJD').text
        
        data.append({
            'Event': event,
            'Edition': edition,
            'Position': position,
            'Athlete': athlete,
            'Country': country,
            'Mark': mark
        })
    
    return pd.DataFrame(data)

def scrape_results(verify_ssl=True):
    # Create directories if they don't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('final_results'):
        os.makedirs('final_results')
        
    # Step 1: Get all championships
    df_championships = first_page_scrape(verify_ssl=verify_ssl)
    
    # Step 2: Get finals URLs for each championship
    for idx, row in df_championships.iterrows():
        results_path = f'results/results_{idx}.csv'
        if not os.path.isfile(results_path):
            print(f"Scraping finals for {row['Meeting']}")
            second_page(row['URL'], results_path, verify_ssl=verify_ssl)
            time.sleep(1)  # Be nice to the server
    
    # Step 3: Get detailed results for each final
    all_results = []
    results_files = os.listdir('results')
    
    for file in results_files:
        df = pd.read_csv(f'results/{file}')
        for idx, row in df.iterrows():
            print(f"Scraping results for {row['Event']}")
            results_df = check_url(row['URL'], verify_ssl=verify_ssl)
            if results_df is not None:
                results_df.to_csv(f'final_results/final_{file}_{idx}.csv', index=False)
                all_results.append(results_df)
            time.sleep(1)  # Be nice to the server
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv('world_athletics_all_results.csv', index=False)
    return final_df

# Run the scraper only when this script is run directly
if __name__ == "__main__":
    # Set verify_ssl=False if you're having SSL certificate issues
    results = scrape_results(verify_ssl=True)
