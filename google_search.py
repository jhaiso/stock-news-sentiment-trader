"""_summary_
    Script that googles a company's name and returns the date-time,
    headline, author, and link of the most recent of the 100 most
    relevant pages if within a certain amount of time beforehand.
Returns:
    dictionary with keys:
        published:  datetime of page publish
        title:      title of the webpage in format HEADLINE-AUTHOR
        link:       link to webpage
"""

import feedparser
import urllib.parse
from time import mktime
from datetime import datetime, timedelta

def fetch_google_news_feed_sorted(query, max_results=10, minutes_back=15):
    q = urllib.parse.quote(query)
    feed_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)

    cutoff_time = datetime.utcnow() - timedelta(minutes=minutes_back)

    # Limit to top 100 results and filter out entries without a date
    entries_with_dates = [
        e for e in feed.entries[:100]
        if hasattr(e, 'published_parsed') and
           datetime.fromtimestamp(mktime(e.published_parsed)) >= cutoff_time
    ]

    # Sort by date, newest first
    sorted_entries = sorted(
        entries_with_dates,
        key=lambda entry: entry.published_parsed,
        reverse=True
    )

    return [{
        "title": entry.title,
        "link": entry.link,
        "published": datetime.fromtimestamp(mktime(entry.published_parsed)).isoformat()
    } for entry in sorted_entries[:max_results]]

if __name__ == "__main__":
    articles = fetch_google_news_feed_sorted("Nuvve Holding", max_results=10, minutes_back=15000)
    for art in articles:
        print(f"• {art['published']} | {art['title']}\n  {art['link']}\n")

