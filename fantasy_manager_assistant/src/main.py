from scraper.news_scraper import ScraperRunner, FutbolFantasyScraper, DataStorage
from entities.ner_models import NERModel


def run_scraper(fetch_content=True):
    ff_scraper = FutbolFantasyScraper()
    storage = DataStorage()
    scraper = ScraperRunner(ff_scraper, storage)
    news_items = scraper.run_single(fetch_content=fetch_content)
    ner = NERModel("Davlan/bert-base-multilingual-cased-ner-hrl")
    if news_items:
        print(f"\nFound {len(news_items)} news items:")
        for i, item in enumerate(news_items, 1):
            print(f"\nItem {i}:")
            print(f"Title: {item.title}")
            print(f"Date: {item.date}")
            print(f"Link: {item.link}")
            print(f"Image: {item.image_url}")

            print(f"Content preview: {item.content}...")
            print("Named entities ", ner.predict([item.title]))
    else:
        print("No news items found. Check the logs for details.")


if __name__ == "__main__":
    run_scraper()
