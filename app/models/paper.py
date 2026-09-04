from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a section in a research paper with title and page boundaries."""

    title: str
    page_start: int
    page_end: int


class Paper(BaseModel):
    """Represents a research paper with metadata and structural sections."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    source_path: str = ""
    sections: list[Section] = Field(default_factory=list)
