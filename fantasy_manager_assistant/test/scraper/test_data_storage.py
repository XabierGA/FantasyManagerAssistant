import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scraper.news_data import NewsArticle
from scraper.data_storage import DataStorage

import shutil
from datetime import datetime
import pandas as pd
from unittest.mock import patch
import json


class TestDataStorage(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "test_scraped_data"
        self.storage = DataStorage(data_dir=self.test_dir)
        self.sample_news_items = [
            NewsArticle(
                title="Test Title 1",
                link="http://test.com/1",
                image_url="http://test.com/1",
                date="2025-01-01",
                category="test",
            ),
            NewsArticle(
                title="Test Title 2",
                link="http://test.com/2",
                image_url="http://test.com/1",
                date="2025-01-02",
                category="test",
            ),
        ]

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_creates_directory(self):
        """Test that the directory is created on initialization."""
        self.assertTrue(os.path.exists(self.test_dir))

    def test_save_to_csv_empty_list(self):
        """Test saving an empty list of news items to CSV."""
        filepath = self.storage.save_to_csv([])
        self.assertEqual(filepath, "")

    def test_save_to_csv_with_custom_filename(self):
        """Test saving news items to CSV with a custom filename."""
        filename = "custom_test_file.csv"
        filepath = self.storage.save_to_csv(self.sample_news_items, filename)

        expected_path = os.path.join(self.test_dir, filename)
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(filepath))

        df = pd.read_csv(filepath)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["title"], "Test Title 1")
        self.assertEqual(df.iloc[1]["link"], "http://test.com/2")

    def test_save_to_csv_with_default_filename(self):
        """Test saving news items to CSV with the default filename."""
        with patch("src.scraper.data_storage.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            mock_datetime.strftime = datetime.strftime

            filepath = self.storage.save_to_csv(self.sample_news_items)

            expected_filename = f"futbol_fantasy_news_20250317_120000.csv"
            expected_path = os.path.join(self.test_dir, expected_filename)
            self.assertEqual(filepath, expected_path)
            self.assertTrue(os.path.exists(filepath))

    def test_save_to_csv_updates_existing_file(self):
        """Test updating an existing CSV file with new items."""
        initial_filepath = self.storage.save_to_csv(
            self.sample_news_items, "test_update.csv"
        )

        new_items = [
            NewsArticle(
                title="Test Title 1",
                link="http://test.com/1",
                image_url="http://test.com/1",
                date="2025-01-01",
                category="test",
            ),
            NewsArticle(
                title="Test Title 3",
                link="http://test.com/3",
                image_url="http://test.com/1",
                date="2025-01-03",
                category="test",
            ),
        ]

        updated_filepath = self.storage.save_to_csv(new_items, "test_update.csv")

        self.assertEqual(updated_filepath, initial_filepath)
        df = pd.read_csv(updated_filepath)

        self.assertEqual(len(df), 3)

        self.assertTrue(any(df["title"] == "Test Title 3"))

    def test_save_to_csv_handles_errors_on_update(self):
        """Test error handling when updating an existing CSV file."""
        invalid_filepath = os.path.join(self.test_dir, "invalid.csv")
        with open(invalid_filepath, "w") as f:
            f.write("This is not a valid CSV file")

        filepath = self.storage.save_to_csv(self.sample_news_items, "invalid.csv")

        self.assertEqual(filepath, invalid_filepath)
        self.assertTrue(os.path.exists(filepath))

        df = pd.read_csv(filepath)
        self.assertEqual(len(df), 2)

    def test_save_to_json_empty_list(self):
        """Test saving an empty list of news items to JSON."""
        filepath = self.storage.save_to_json([])
        self.assertEqual(filepath, "")

    def test_save_to_json_with_custom_filename(self):
        """Test saving news items to JSON with a custom filename."""
        filename = "custom_test_file.json"
        filepath = self.storage.save_to_json(self.sample_news_items, filename)

        expected_path = os.path.join(self.test_dir, filename)
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(filepath))

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["title"], "Test Title 1")
        self.assertEqual(data[1]["link"], "http://test.com/2")

    def test_save_to_json_with_default_filename(self):
        """Test saving news items to JSON with the default filename."""
        with patch("src.scraper.data_storage.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 17, 12, 0, 0)
            mock_datetime.strftime = datetime.strftime

            filepath = self.storage.save_to_json(self.sample_news_items)

            expected_filename = f"futbol_fantasy_news_20250317_120000.json"
            expected_path = os.path.join(self.test_dir, expected_filename)
            self.assertEqual(filepath, expected_path)
            self.assertTrue(os.path.exists(filepath))

    def test_save_to_json_content_format(self):
        """Test that JSON content is formatted correctly with proper indentation."""
        filepath = self.storage.save_to_json(self.sample_news_items, "format_test.json")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('"title":"Test Title 1"', content)

        self.assertFalse(r"\u" in content)


if __name__ == "__main__":
    unittest.main()
