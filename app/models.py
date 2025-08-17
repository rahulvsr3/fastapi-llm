from pydantic import BaseModel

class SentimentRequest(BaseModel):
    keyword: str

class SentimentResponse(BaseModel):
    keyword: str
    sentiment_summary: str
