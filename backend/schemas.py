from pydantic import BaseModel
from typing import List, Optional


class Source(BaseModel):
    """Metadata for a retrieved document chunk."""
    doc_id: str
    filename: str
    page: Optional[int] = None
    score: Optional[float] = None
    snippet: Optional[str] = None
    domain: str = "unknown"


class RAGAnswer(BaseModel):
    """Structured JSON response returned by the RAG pipeline."""
    answer: str                      # The final generated answer
    confidence: float                # Confidence score (0.0–1.0)
    missing_info: List[str]          # List of gaps or uncertainties
    citations: List[Source]          # Supporting document references
    reasoning_summary: Optional[str] = None  # How the model reasoned
    suggestions: List[str] = []      # Enrichment suggestions
