import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/cointegrated/rubert-tiny2-cedr-emotion-detection"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query_emotion(text):
    """Send text to Hugging Face model and return list of emotions with scores"""
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def get_dominant_emotion(text):
    """Return the emotion label with highest probability"""
    output = query_emotion(text)
    if not output or not isinstance(output[0], list):
        return None
    emotions = output[0]
    best = max(emotions, key=lambda x: x["score"])
    return best["label"]

print(get_dominant_emotion("I am very happy today!")) 