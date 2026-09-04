from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a text chunk extracted from a paper for retrieval."""

    chunk_id: str
    paper_id: str
    page: int
    section: str
    text: str


class Evidence(Chunk):
    """Represents retrieved evidence from a chunk with source identity and score."""

    source_id: str
    score: float | None = None
