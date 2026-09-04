from pydantic import BaseModel, Field

from app.models.chunk import Evidence


class Citation(BaseModel):
    """Maps a source ID to paper title, page, and section provenance."""

    source_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str


class ResearchResult(BaseModel):
    """Represents the final answer, citations, sources, and research trace."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)
    research_trace: list[str] = Field(default_factory=list)
