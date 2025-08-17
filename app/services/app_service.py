import requests
from app.config import TWITTER_BEARER_TOKEN

TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

def fetch_tweets(keyword: str, max_results: int = 10):
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    params = {"query": keyword, "max_results": max_results, "tweet.fields": "text"}
    
    response = requests.get(TWITTER_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    tweets = response.json().get("data", [])
    return [tweet["text"] for tweet in tweets]
