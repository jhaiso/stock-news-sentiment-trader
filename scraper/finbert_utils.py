from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(text: str) -> Tuple[float, str]:
    if not text:
        return 0, labels[-1]

    # Tokenize the full text without truncating
    tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    input_ids = tokens["input_ids"][0]  # remove batch dimension
    chunk_size = 512

    # Split input_ids into chunks of 512
    chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

    # Track sum of softmax scores
    sentiment_scores = torch.zeros(len(labels), device=device)

    for chunk in chunks:
        inputs = {
            "input_ids": chunk.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(chunk).unsqueeze(0).to(device),
        }
        with torch.no_grad():
            logits = model(**inputs)["logits"]
            probs = torch.nn.functional.softmax(logits[0], dim=-1)
            sentiment_scores += probs

    # Average sentiment across chunks
    sentiment_scores /= len(chunks)

    probability = torch.max(sentiment_scores)
    sentiment = labels[torch.argmax(sentiment_scores)]

    return probability.item(), sentiment

if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment('Nuvve Holding ( (NVVE) ) has issued an update.\nOn May 9, 2025, Nuvve Holding Corp. announced its engagement with multiple digital asset advisory consultants to accelerate the growth of its new subsidiary, Nuvve-DigitalAssets.\nThis strategic move aims to enhance Nuvve’s digital asset portfolio and create long-term shareholder value through blockchain innovation.\nThe company has formed a Digital Asset Management Portfolio Committee, chaired by renowned crypto investor James Altucher, to oversee investment decisions.\nMore about Nuvve Holding Nuvve Holding Corp. (NASDAQ: NVVE) is a global leader in vehicle-to-grid (V2G) technology, which enables electric vehicles to store and discharge energy, transforming them into mobile energy resources to help stabilize the grid.')
    print(tensor, sentiment)