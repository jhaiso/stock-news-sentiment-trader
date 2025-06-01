# gpt_utils.py

import os
import json
import openai

# 1) load any .env in your CWD
from dotenv import load_dotenv
load_dotenv()               # will by default look for a file called .env

# 2) pull the key out of the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env file")

def estimate_sentiment(text: str) -> tuple[float, str]:
    """
    Classify the given financial text as Positive/Neutral/Negative
    with probabilities. Returns (best_prob, best_label).
    Uses gpt-4o-mini via OpenAI ChatCompletion.
    """
    system_prompt = (
        "You are a Financial Sentiment Classifier.\n"
        "When given a short passage about a company, market or earnings:\n"
        "1. Respond with EXACTLY one JSON object and nothing else.\n"
        "2. The object must have three keys in this order: \"positive\", \"neutral\", \"negative\".\n"
        "3. All three values must sum exactly to 1.000.\n"
        "4. Do not add any commentary, prefixes, or markdown.\n"
    )
    user_prompt = f'Text: "{text.strip()}"\n\nRespond with JSON.'

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature=0.01,
        max_tokens=150,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content

    try:
        sentiment = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON:\n{raw}\nError: {e}")

    for k in ("positive", "neutral", "negative"):
        if k not in sentiment:
            raise KeyError(f"Missing '{k}' in response: {sentiment}")
        sentiment[k] = float(sentiment[k])

    total = sum(sentiment.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"Probabilities must sum to 1. Got {total}")

    best_label = max(sentiment, key=sentiment.get)
    best_prob  = sentiment[best_label]
    return best_prob, best_label


if __name__ == "__main__":
    txt = (
        "On May 9, 2025, Nuvve Holding Corp. announced its engagement with "
        "multiple digital asset advisory consultants to accelerate the growth "
        "of its new subsidiary, Nuvve-DigitalAssets..."
    )
    print(estimate_sentiment(txt))