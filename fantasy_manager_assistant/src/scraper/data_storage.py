from scraper.news_data import NewsArticle
from typing import Optional, List
import os
from datetime import datetime
import pandas as pd
from dataclasses import asdict
from loggers.main_logger import logger


class DataStorage:
    """Class for handling data storage operations."""

    def __init__(self, data_dir: str = "scraped_data"):
        """Initialize the data storage.

        Args:
            data_dir: Directory to store scraped data.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_to_csv(
        self, news_items: List[NewsArticle], filename: Optional[str] = None
    ) -> str:
        """Save news items to a CSV file.

        Args:
            news_items: List of NewsArticle objects to save.
            filename: Optional filename for the CSV file.

        Returns:
            The path to the saved CSV file.
        """
        if not news_items:
            logger.warning("No news items to save")
            return ""

        if filename is None:
            filename = (
                f"futbol_fantasy_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        filepath = os.path.join(self.data_dir, filename)

        data = [asdict(item) for item in news_items]
        df = pd.DataFrame(data)

        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(filepath)
                combined_df = pd.concat([existing_df, df]).drop_duplicates(
                    subset=["title", "link"]
                )
                combined_df.to_csv(filepath, index=False)
                logger.info(
                    f"Updated existing file {filepath} with {len(df)} new items"
                )
            except Exception as e:
                logger.error(f"Error updating existing file: {e}")
                df.to_csv(filepath, index=False)
                logger.info(f"Created new file {filepath} with {len(df)} items")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"Created new file {filepath} with {len(df)} items")

        return filepath

    def save_to_json(
        self, news_items: List[NewsArticle], filename: Optional[str] = None
    ) -> str:
        """Save news items to a JSON file.

        Args:
            news_items: List of NewsArticle objects to save.
            filename: Optional filename for the JSON file.

        Returns:
            The path to the saved JSON file.
        """
        if not news_items:
            logger.warning("No news items to save")
            return ""

        if filename is None:
            filename = (
                f"futbol_fantasy_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = os.path.join(self.data_dir, filename)

        data = [asdict(item) for item in news_items]
        df = pd.DataFrame(data)

        df.to_json(filepath, orient="records", force_ascii=False, indent=4)
        logger.info(f"Saved {len(df)} items to JSON file {filepath}")

        return filepath
