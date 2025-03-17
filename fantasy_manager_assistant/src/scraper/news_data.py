from dataclasses import dataclass
from datetime import datetime


@dataclass
class NewsArticle:
    """Data class representing a news article."""

    title: str
    link: str
    date: str
    image_url: str
    category: str = ""
    author: str = ""
    content: str = ""
    scraped_at: str = ""

    def __post_init__(self):
        """Set default scraped_at time if not provided."""
        if not self.scraped_at:
            self.scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
