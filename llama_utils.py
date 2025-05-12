# llama_utils.py

import os
import time
import requests
import json
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
#  Load .env (must contain LLAMA_API_KEY) and never commit .env to VCS!
load_dotenv()
API_KEY = os.getenv("LLAMA_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set LLAMA_API_KEY in your .env file")
# ──────────────────────────────────────────────────────────────────────────────

# Dartmouth endpoints
_JWT_URL  = "https://api.dartmouth.edu/api/jwt"
_CHAT_URL = "https://api.dartmouth.edu/api/ai/tgi/codellama-13b-instruct-hf/v1/chat/completions"

# module-level cache for JWT
_jwt_token      = None
_jwt_expires_at = 0

def _get_jwt() -> str:
    """
    Fetch (or return cached) JWT for Dartmouth AI.
    Assumes token lives ~1 hour; refreshes 5 minutes before expiry.
    """
    global _jwt_token, _jwt_expires_at
    now = time.time()
    if _jwt_token and now < _jwt_expires_at - 300:
        return _jwt_token

    resp = requests.post(_JWT_URL, headers={"Authorization": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    token = data.get("jwt")
    if not token:
        raise RuntimeError("Failed to retrieve JWT: " + json.dumps(data))
    _jwt_token      = token
    _jwt_expires_at = now + 3600
    return token

def estimate_sentiment(text: str) -> tuple[float, str]:
    """
    Classify the given financial text as Positive/Neutral/Negative
    with probabilities. Returns (best_prob, best_label).
    """
    jwt = _get_jwt()

    system_prompt = (
        "You are a Financial Sentiment Classifier.\n"
        "When given a short passage about a company, market or earnings:\n"
        "1. Respond with EXACTLY one JSON object and nothing else.\n"
        "2. The object must have three keys in this order: \"positive\", \"neutral\", \"negative\".\n"
        "3. All three values must sum exactly to 1.000.\n"
        "4. Do not add any commentary, prefixes, or markdown.\n\n"
    )
    user_prompt = f"Text: \"{text.strip()}\"\n\nRespond with JSON."

    payload = {
        "model": "codellama-13b-instruct-hf",
        "temperature": 0.01,
        "max_output_tokens": 150,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type":  "application/json"
    }

    resp = requests.post(_CHAT_URL, json=payload, headers=headers)
    if not resp.ok:
        print(f"ERROR {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    result = resp.json()
    try:
        raw = result["choices"][0]["message"]["content"]
        sentiment = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse response: {e}\nFull response: {result}")

    # Validate and pick best
    for k in ("positive", "neutral", "negative"):
        if k not in sentiment:
            raise KeyError(f"Missing '{k}' in model output: {sentiment}")
        sentiment[k] = float(sentiment[k])

    best_label = max(sentiment, key=sentiment.get)
    best_prob  = sentiment[best_label]
    return best_prob, best_label

if __name__ == "__main__":
    txt = (
        "On May 9, 2025, Nuvve Holding Corp. announced its engagement with multiple "
        "digital asset advisory consultants to accelerate the growth of its new "
        "subsidiary, Nuvve-DigitalAssets. This strategic move aims to enhance "
        "Nuvve’s digital asset portfolio and create long-term shareholder value..."
    )
    print(estimate_sentiment(txt))