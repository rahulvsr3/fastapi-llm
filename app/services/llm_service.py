from app.services import openai_utils
import openai

def analyze_sentiment(tweets):
    results = []
    if not tweets:
        return {"total_tweets": 0, "positive": 0, "negative": 0, "results": []}

    for tweet in tweets:
        tokens_needed = len(tweet.split())
        if not openai_utils.can_use_tokens(tokens_needed):
            results.append({"tweet": tweet, "sentiment": "SKIPPED - token limit reached"})
            continue

        prompt = f"Analyze the sentiment of the following tweet. Reply only with POSITIVE or NEGATIVE.\n\nTweet: {tweet}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        sentiment = response.choices[0].message["content"].strip().upper()
        results.append({"tweet": tweet, "sentiment": sentiment})

        openai_utils.update_token_usage(tokens_needed)

    positive = sum(1 for r in results if r['sentiment'] == "POSITIVE")
    negative = sum(1 for r in results if r['sentiment'] == "NEGATIVE")

    return {
        "total_tweets": len(tweets),
        "positive": positive,
        "negative": negative,
        "results": results
    }
