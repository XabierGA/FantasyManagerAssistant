import re
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from scraper.news_data import NewsArticle
from scraper.data_storage import DataStorage
from loggers.main_logger import logger
import requests
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag


class WebScraper:
    """Base class for web scrapers with common functionality."""

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        """Initialize the web scraper.

        Args:
            base_url: The base URL of the website to scrape.
            headers: Optional request headers.
        """
        self.base_url = base_url
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch the HTML content of a page.

        Args:
            url: The URL to fetch.

        Returns:
            The HTML content of the page or None if the request failed.
        """
        try:
            logger.info(f"Fetching page: {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def normalize_url(self, url: str) -> str:
        """Normalize a URL by ensuring it has the correct scheme and domain.

        Args:
            url: The URL to normalize.

        Returns:
            The normalized URL.
        """
        if not url:
            return ""

        if url.startswith("http"):
            return url

        return (
            self.base_url.rstrip("/") + ("/" if not url.startswith("/") else "") + url
        )


class FutbolFantasyScraper(WebScraper):
    """Scraper specialized for the Futbol Fantasy website."""

    def __init__(self, base_url: str = "https://www.futbolfantasy.com/"):
        """Initialize the Futbol Fantasy scraper.

        Args:
            base_url: The base URL of the Futbol Fantasy website.
        """
        super().__init__(base_url)
        self.categories = {
            "laliga": "LaLiga",
            "champions": "Champions League",
            "selecciones": "Selecciones",
            "fichajes": "Fichajes",
            "internacional": "Internacional",
        }

    def extract_date(self, container: Tag) -> str:
        """Extract the date from a news container.

        Args:
            container: The news container element.

        Returns:
            The extracted date or "No date" if not found.
        """
        # First, try to find a direct date element using regex pattern
        date_pattern = re.compile(r"\d+\s+\w+")
        date_text = container.find(text=date_pattern)

        if date_text:
            return date_text.strip()

        # If that fails, look for an element with the date class
        for date_class in [".date", ".time", ".published", ".post-date"]:
            date_element = container.select_one(date_class)
            if date_element:
                return date_element.text.strip()

        return "No date"

    def extract_image_url(self, container: Tag) -> str:
        """Extract the image URL from a news container.

        Args:
            container: The news container element.

        Returns:
            The extracted image URL or an empty string if not found.
        """
        img_element = container.select_one("img")
        if img_element:
            return self.normalize_url(img_element.get("src", ""))
        return ""

    def extract_category_from_url(self, url: str) -> str:
        """Extract the category from a news article URL.

        Args:
            url: The URL of the news article.

        Returns:
            The extracted category or an empty string if not found.
        """
        if not url:
            return ""

        for category_key, category_name in self.categories.items():
            if f"/{category_key}/" in url:
                return category_name

        return ""

    def parse_news_items(self, html_content: str) -> List[NewsArticle]:
        """Extract news items from HTML content.

        Args:
            html_content: The HTML content to parse.

        Returns:
            A list of NewsArticle objects.
        """
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        news_articles = []


        news_containers = soup.select("div.noticia-container")
        logger.info(f"Found {len(news_containers)} news containers")

        for container in news_containers:
            try:
                news_link = container.select_one("a.noticia")

                if not news_link:
                    continue

                title = news_link.get("alt", "").strip()
                if not title:
                    title = news_link.text.strip()

                link = self.normalize_url(news_link.get("href", ""))

                date = self.extract_date(container)

                image_url = self.extract_image_url(container)

                category = self.extract_category_from_url(link)

                article = NewsArticle(
                    title=title,
                    link=link,
                    date=date,
                    image_url=image_url,
                    category=category,
                )

                news_articles.append(article)

            except Exception as e:
                logger.error(f"Error parsing news item: {e}")
                continue

        return news_articles

    def fetch_article_content(self, article: NewsArticle) -> str:
        """Fetch and extract the full content of an article.

        Args:
            article: The NewsArticle object to update with content.

        Returns:
            The extracted article content or an empty string if not found.
        """
        if not article.link:
            return ""

        html_content = self.fetch_page(article.link)
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")

        content_selectors = [
            "div.article-content",
            "div.entry-content",
            "div.post-content",
            "div.content",
            "div.noticia-contenido",
            "article",
        ]

        for selector in content_selectors:
            content_container = soup.select_one(selector)
            if content_container:
                paragraphs = content_container.select("p")
                if paragraphs:
                    content = "\n\n".join([p.text.strip() for p in paragraphs])
                    return content

        paragraphs = soup.select("p")
        if paragraphs:
            substantial_paragraphs = [
                p.text.strip() for p in paragraphs if len(p.text.strip()) > 60
            ]
            if substantial_paragraphs:
                return "\n\n".join(substantial_paragraphs)

        return ""


class ScraperRunner:
    """Class for managing and running scraper operations."""

    def __init__(self, scraper: FutbolFantasyScraper, storage: DataStorage):
        """Initialize the scraper runner.

        Args:
            scraper: The scraper instance to use.
            storage: The data storage instance to use.
        """
        self.scraper = scraper
        self.storage = storage

    def run_single(
        self, fetch_content: bool = False, save_format: str = "csv"
    ) -> List[NewsArticle]:
        """Run the scraper once.

        Args:
            fetch_content: Whether to fetch the full content of each article.
            save_format: The format to save the data in ('csv' or 'json').

        Returns:
            A list of scraped NewsArticle objects.
        """
        logger.info(f"Starting scraper for {self.scraper.base_url}")
        html_content = self.scraper.fetch_page(self.scraper.base_url)

        if not html_content:
            logger.error("Failed to fetch the page")
            return []

        news_items = self.scraper.parse_news_items(html_content)
        logger.info(f"Found {len(news_items)} news items")

        if fetch_content and news_items:
            logger.info("Fetching full article content...")
            for item in news_items:
                content = self.scraper.fetch_article_content(item)
                if content:
                    item.content = content

                time.sleep(0.1)

        if news_items:
            if save_format.lower() == "json":
                self.storage.save_to_json(news_items)
            else:
                self.storage.save_to_csv(news_items)

        return news_items

    def run_scheduled(
        self,
        interval_minutes: int = 60,
        fetch_content: bool = False,
        save_format: str = "csv",
    ):
        """Run the scraper on a schedule.

        Args:
            interval_minutes: The interval between scraper runs in minutes.
            fetch_content: Whether to fetch the full content of each article.
            save_format: The format to save the data in ('csv' or 'json').
        """
        logger.info(
            f"Starting scheduled scraper with {interval_minutes} minute interval"
        )

        while True:
            try:
                self.run_single(fetch_content=fetch_content, save_format=save_format)
                logger.info(
                    f"Sleeping for {interval_minutes} minutes before next scrape"
                )
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Scraper stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.info("Retrying in 5 minutes")
                time.sleep(300)
