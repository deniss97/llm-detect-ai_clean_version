from datetime import datetime

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    text: str


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DetectResponse(BaseModel):
    id: int
    text: str
    ai_probability: float
    ai_percent: float
    verdict: str
    confidence: float
    created_at: datetime


class ResultResponse(DetectResponse):
    source: str


class HealthResponse(BaseModel):
    status: str
    detector_mode: str
