from fastapi import FastAPI
from pydantic import BaseModel
from app.services import app_service, llm_service, openai_utils

app = FastAPI(title="FastAPI LLM Sentiment API (OpenAI)")

class SentimentRequest(BaseModel):
    keyword: str
    limit: int = 10

@app.get("/")
def root():
    return {"message": "Hello, FastAPI with OpenAI GPT-3.5 sentiment analysis!"}

@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest):
    tweets = app_service.fetch_tweets(request.keyword, request.limit)
    sentiment_summary = llm_service.analyze_sentiment(tweets)
    return {
        "keyword": request.keyword,
        "tweets": tweets,
        "sentiment_summary": sentiment_summary
    }

@app.get("/openai-usage")
def openai_usage():
    used = openai_utils.read_token_usage()
    remaining = max(0, openai_utils.MAX_TOKENS_PER_MONTH - used)
    return {"tokens_used": used, "tokens_remaining": remaining}
