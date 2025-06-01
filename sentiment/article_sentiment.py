from newspaper import Article
import logging

# Suppress the newspaper library's logging output
logging.getLogger("newspaper").setLevel(logging.CRITICAL)

def extract_main_content(url):
    try:
        article = Article(url)
        article.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        article.download()
        article.parse()
        article.nlp()

        return {
            "headline": article.title,
            "body_text": article.text,
            "summary": article.summary
        }

    except Exception as e:
        print(f"[ERROR] Failed to download or parse the article at {url}: {e}")
        return None

if __name__ == "__main__":
    # Example URL (Replace with actual URLs from your earlier search)
    url = "https://www.tipranks.com/news/company-announcements/nuvve-holding-engages-advisors-for-digital-asset-growth"
    
    content = extract_main_content(url)
    print(content)
