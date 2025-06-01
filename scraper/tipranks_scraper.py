import re
import json
import time
import csv
import requests
from bs4 import BeautifulSoup
from finbert_utils import estimate_sentiment
# from llama_utils import estimate_sentiment

# 1) Original news page is the SPA that contains the JSON blob
NEWS_SPA_URL = "https://www.tipranks.com/news"
# 2) Article pages are on the blog subdomain
BLOG_BASE    = "https://blog.tipranks.com"

def fetch_state_json(url):
    """
    Fetch the SPA page and parse out window.__STATE__.
    Handles both JSON.parse("…") and direct object assignment.
    Returns the parsed JSON dict, or None on failure.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"✖ Failed to fetch SPA JSON at {url!r}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for script in soup.find_all("script"):
        src = script.string or script.text or ""
        if "window.__STATE__" not in src:
            continue

        # 1) Try JSON.parse("…");
        m = re.search(r'window\.__STATE__\s*=\s*JSON\.parse\("([\s\S]*?)"\);', src)
        if m:
            js_escaped = m.group(1)
            decoded = js_escaped.encode("utf-8").decode("unicode_escape")
            try:
                return json.loads(decoded)
            except json.JSONDecodeError as e:
                print(f"  ⚠ JSON.parse decode error: {e}")
                return None

        # 2) Fallback: direct = {…}
        m2 = re.search(r'window\.__STATE__\s*=\s*({[\s\S]*?})', src)
        if m2:
            try:
                return json.loads(m2.group(1))
            except json.JSONDecodeError as e:
                print(f"  ⚠ direct JSON decode error: {e}")
                return None

    print("✖ Did not find any window.__STATE__ in SPA")
    return None

def build_blog_url(post):
    """
    The blog URL is https://blog.tipranks.com/YYYY/MM/DD/<slug>/
    post['date'] is ISO like "2025-05-09T18:49:58.000Z"
    and post['slug'] is the article slug.
    """
    date = post["date"][:10]  # "YYYY-MM-DD"
    slug = post["slug"]
    return f"{BLOG_BASE}/{date}/{slug}/"

def scrape_text(url):
    """
    Fetch the blog URL and pull out only the <p> text inside
    <div class="post-entry">…</div>. Returns the cleaned text,
    or None on failure.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"    ✖ Failed to fetch article at {url!r}: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # 1) Locate the post-entry container
    container = soup.find("div", class_="post-entry")
    if not container:
        print(f"    ⚠ No <div class='post-entry'> found in {url!r}")
        return None

    # 2) Extract only its <p> children
    paras = container.find_all("p")
    if not paras:
        print(f"    ⚠ No <p> tags inside post-entry for {url!r}")
        return None

    # 3) Join and return their text content
    lines = []
    for p in paras:
        text = p.get_text(separator=" ", strip=True)
        # skip empty strings
        if text:
            lines.append(text)
    return "\n\n".join(lines)

def run_trending_blog_pipeline():
    print("1) Fetching SPA JSON from:", NEWS_SPA_URL)
    state = fetch_state_json(NEWS_SPA_URL)
    if state is None:
        print("❌ Cannot continue without SPA JSON, exiting.")
        return

    trending = state.get("MainNews", {})\
                    .get("posts", {})\
                    .get("trending", [])
    print(f"2) Found {len(trending)} trending items")

    with open("trending_blog.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["url", "tickers_from_state", "sentiment", "score"])

        for post in trending:
            url = build_blog_url(post)
            print("→", url)

            #  Extract tickers straight from the JSON 'stocks' array
            page_tickers = [
                s["ticker"]
                for s in post.get("stocks", [])
                if s.get("ticker")
            ]

            # scrape the blog page for its paragraphs (text only)
            text = scrape_text(url)
            if text is None:
                print("    → skipping due to fetch error.")
                continue

            # sentiment
            try:
                score, label = estimate_sentiment(text)
            except Exception as e:
                print(f"    ✖ Sentiment analysis failed on {url!r}: {e}")
                continue

            # write to CSV
            writer.writerow([
                url,
                "|".join(page_tickers),
                label,
                f"{score:.4f}"
            ])

            print(f"   [{label} {score:.2f}] tickers: {page_tickers}")
            time.sleep(1)  # polite crawl delay

if __name__ == "__main__":
    run_trending_blog_pipeline()