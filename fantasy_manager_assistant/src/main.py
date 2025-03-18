from scraper.news_scraper import ScraperRunner, FutbolFantasyScraper, DataStorage
from entities.ner_models import SentimentdNERModel


def run_scraper(fetch_content=True):
    ff_scraper = FutbolFantasyScraper()
    storage = DataStorage()
    scraper = ScraperRunner(ff_scraper, storage)
    news_items = scraper.run_single(fetch_content=fetch_content)
    ner = SentimentdNERModel(
    model_name="Davlan/bert-base-multilingual-cased-ner-hrl",
    sentiment_model="tabularisai/multilingual-sentiment-analysis"
)
    if news_items:
        print(f"\nFound {len(news_items)} news items:")
        for i, item in enumerate(news_items, 1):
            print(f"\nItem {i}:")
            print(f"Title: {item.title}")
            print(f"Date: {item.date}")
            print(f"Link: {item.link}")
            print(f"Image: {item.image_url}")

            print(f"Content preview: {item.content}...")
            print("Named entities ", ner.predict_with_sentiment([item.content.replace("CEO y administrador de FutbolFantasy.com desde 2011. Programador informático y desarrollador de aplicaciones multiplataforma. Redactor jefe, community manager y streamer.","")]))
    else:
        print("No news items found. Check the logs for details.")


if __name__ == "__main__":
    run_scraper()
