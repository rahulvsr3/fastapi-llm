from fastapi import FastAPI, HTTPException
from app.models import SentimentRequest, SentimentResponse
from app.services import twitter_service, llm_service

app = FastAPI(title="Twitter Sentiment Analysis API (Free LLM, Background Download)")

# Start model download in background when app starts
@app.on_event("startup")
def startup_event():
    llm_service.start_model_download_background()

@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(req: SentimentRequest):
    try:
        tweets = twitter_service.fetch_tweets(req.keyword)
        if not tweets:
            return {"keyword": req.keyword, "sentiment_summary": "No tweets found for this keyword."}
        
        summary = llm_service.analyze_sentiment(tweets)
        return {"keyword": req.keyword, "sentiment_summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    """Returns LLM model download, storage, and queue status"""
    try:
        status_info = llm_service.get_model_status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
