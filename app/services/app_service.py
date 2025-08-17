import snscrape.modules.twitter as sntwitter

def fetch_tweets(keyword: str, limit: int = 10):
    """
    Fetch recent tweets using snscrape (no API key required).
    Returns a list of tweet texts.
    """
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if i >= limit:
            break
        tweets.append(tweet.content)
    return tweets
