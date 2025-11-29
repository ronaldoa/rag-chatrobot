"""API data models."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Chat request."""

    message: str = Field(..., description="User question", min_length=1)
    include_sources: bool = Field(True, description="Return sources or not")


class Source(BaseModel):
    """Document source."""

    content: str
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response."""

    answer: str
    sources: Optional[List[Source]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
