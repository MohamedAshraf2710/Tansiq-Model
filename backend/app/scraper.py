"""
scraper.py — Dynamic data handler with DB-first lookup and web scraping fallback.
Provides faculty details from the database, falling back to fast async
web scraping when the requested data is not available locally.
If a faculty URL is known, it scrapes that URL directly.
"""

import logging
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from app.config import settings
from app.database import engine

logger = logging.getLogger(__name__)

# Timeout for web requests — maximum speed
_REQUEST_TIMEOUT = 5
_MAX_WORKERS = 4

# ======================================================================
# DB-first Lookup
# ======================================================================

def get_faculty_details(faculty_name: str) -> Optional[dict]:
    if not faculty_name or not faculty_name.strip():
        return None

    try:
        with engine.connect() as conn:
            query = text(
                """
                SELECT faculty, governorate, score, college_vision,
                       url, college_field, boys, girls
                FROM faculties
                WHERE LOWER(faculty) LIKE :name
                LIMIT 1
                """
            )
            result = conn.execute(
                query, {"name": f"%{faculty_name.strip().lower()}%"}
            )
            row = result.fetchone()

            if row:
                return {
                    "faculty": row[0],
                    "governorate": row[1],
                    "score": row[2],
                    "vision": row[3],
                    "url": row[4],
                    "field": row[5],
                    "boys": row[6],
                    "girls": row[7],
                    "source": "database",
                }
    except Exception as e:
        logger.error("DB lookup failed for '%s': %s", faculty_name, e)

    return None

def search_faculties_db(keyword: str, limit: int = 10) -> list[dict]:
    if not keyword or not keyword.strip():
        return []

    try:
        with engine.connect() as conn:
            query = text(
                """
                SELECT faculty, governorate, score, url, college_field
                FROM faculties
                WHERE LOWER(faculty) LIKE :kw
                   OR LOWER(college_field) LIKE :kw
                   OR LOWER(college_vision) LIKE :kw
                ORDER BY score DESC
                LIMIT :lim
                """
            )
            result = conn.execute(
                query,
                {"kw": f"%{keyword.strip().lower()}%", "lim": limit},
            )
            return [
                {
                    "faculty": r[0],
                    "governorate": r[1],
                    "score": r[2],
                    "url": r[3],
                    "field": r[4],
                    "source": "database",
                }
                for r in result
            ]
    except Exception as e:
        logger.error("DB search failed for '%s': %s", keyword, e)
        return []

# ======================================================================
# Web Scraping Fallback — Fast & Concurrent
# ======================================================================

_SEARCH_ENGINES = [
    "https://www.google.com/search?q={query}+كلية+مصر+تنسيق&hl=ar&num=5",
    "https://www.google.com/search?q={query}+حد+أدنى+تنسيق&hl=ar&num=5",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en;q=0.9",
}

def _scrape_direct_url(url: str) -> list[dict]:
    """Scrape the specific faculty URL directly."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract main text from paragraphs
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        
        if text_content:
            return [{"text": text_content[:1000], "url": url, "source": "web_direct"}]
    except Exception as e:
        logger.warning("Direct scrape failed for URL %s: %s", url, e)
    return []

def _scrape_single_url(url: str) -> list[dict]:
    results = []
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select("div.BNeawe, div.s3v9rd, div.kCrYT"):
            text_content = item.get_text(strip=True)
            if len(text_content) > 30:
                link_tag = item.find("a", href=True)
                link = None
                if link_tag:
                    href = link_tag["href"]
                    match = re.search(r"/url\?q=([^&]+)", href)
                    if match:
                        link = match.group(1)
                    elif href.startswith("http"):
                        link = href

                results.append({"text": text_content[:500], "url": link, "source": "web_search"})

                if len(results) >= 3:
                    break
    except Exception as e:
        logger.warning("Search scrape failed for URL: %s", e)

    return results

def scrape_web(query: str, direct_url: Optional[str] = None) -> list[dict]:
    # Try direct URL first if available
    if direct_url and direct_url.startswith("http"):
        direct_results = _scrape_direct_url(direct_url)
        if direct_results:
            return direct_results

    if not query or not query.strip():
        return []

    urls = [u.format(query=requests.utils.quote(query)) for u in _SEARCH_ENGINES]
    all_results = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {executor.submit(_scrape_single_url, url): url for url in urls}
        for future in as_completed(futures, timeout=_REQUEST_TIMEOUT + 2):
            try:
                results = future.result(timeout=_REQUEST_TIMEOUT)
                all_results.extend(results)
            except Exception as e:
                logger.warning("Scrape future failed: %s", e)

    seen = set()
    unique = []
    for r in all_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:5]

# ======================================================================
# Unified Lookup — DB first, Web fallback
# ======================================================================

def smart_lookup(query: str) -> dict:
    # 1. DB search
    db_results = search_faculties_db(query)
    if db_results:
        logger.info("smart_lookup: Found %d results in DB", len(db_results))
        return {"results": db_results, "source": "database"}

    # 2. Specific detail lookup
    faculty_detail = get_faculty_details(query)
    if faculty_detail:
        logger.info("smart_lookup: Found faculty detail in DB")
        return {"results": [faculty_detail], "source": "database"}

    # 3. Web fallback (no direct URL known yet, so pass query)
    logger.info("smart_lookup: DB miss — falling back to web scraping")
    web_results = scrape_web(query)
    if web_results:
        return {"results": web_results, "source": "web"}

    return {"results": [], "source": "none"}
