import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def scrape_faculty_details(url: str) -> str:
    """
    Scrapes the official university website to extract extra context 
    if database features are insufficient.
    """
    if pd.isna(url) or not str(url).startswith('http'):
        return ""
    
    try:
        logger.info(f"DataOps: Fallback Scraper triggered for URL: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # Timeout set to 3 seconds max to prevent blocking the API request
        response = requests.get(url, headers=headers, timeout=3, verify=False)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from the first 3 paragraphs to capture main info
            paragraphs = soup.find_all('p', limit=3)
            extracted_text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Return capped slice to prevent prompt token bloating
            return extracted_text[:600] 
        return ""
    except Exception as e:
        logger.warning(f"Scraper failed to fetch URL {url}: {e}")
        return ""