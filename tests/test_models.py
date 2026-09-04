from app.models import Chunk, Citation, Evidence, Paper, ResearchResult, Section


def test_paper_and_section_construction_and_serialization():
    section = Section(title="Introduction", page_start=1, page_end=2)
    paper = Paper(
        paper_id="paper_001",
        title="Domain Generalization in Face Anti-Spoofing",
        authors=["Alice Smith", "Bob Jones"],
        year=2023,
        abstract="This paper explores domain generalization...",
        source_path="data/papers/paper_001.pdf",
        sections=[section],
    )

    assert paper.paper_id == "paper_001"
    assert paper.sections[0].title == "Introduction"
    assert len(paper.authors) == 2

    json_str = paper.model_dump_json()
    paper_reconstructed = Paper.model_validate_json(json_str)

    assert paper_reconstructed == paper


def test_chunk_and_evidence_construction_and_serialization():
    chunk = Chunk(
        chunk_id="chunk_001",
        paper_id="paper_001",
        page=3,
        section="Methodology",
        text="We propose a novel feature alignment loss.",
    )

    assert chunk.paper_id == "paper_001"
    assert chunk.chunk_id == "chunk_001"
    assert chunk.page == 3
    assert chunk.section == "Methodology"
    assert chunk.text == "We propose a novel feature alignment loss."

    evidence = Evidence(
        source_id="S1",
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        page=chunk.page,
        section=chunk.section,
        text=chunk.text,
        score=0.92,
    )

    assert evidence.source_id == "S1"
    assert evidence.score == 0.92

    json_str = evidence.model_dump_json()
    evidence_reconstructed = Evidence.model_validate_json(json_str)

    assert evidence_reconstructed == evidence


def test_citation_and_research_result_construction_and_serialization():
    citation = Citation(
        source_id="S1",
        paper_id="paper_001",
        paper_title="Domain Generalization in Face Anti-Spoofing",
        page=3,
        section="Methodology",
    )

    assert citation.source_id == "S1"
    assert citation.paper_title == "Domain Generalization in Face Anti-Spoofing"
    assert citation.page == 3
    assert citation.section == "Methodology"

    evidence = Evidence(
        source_id="S1",
        chunk_id="chunk_001",
        paper_id="paper_001",
        page=3,
        section="Methodology",
        text="We propose a novel feature alignment loss.",
        score=0.92,
    )

    result = ResearchResult(
        question="What is the proposed methodology?",
        answer="The proposed methodology uses feature alignment loss [S1].",
        citations=[citation],
        sources=[evidence],
        research_trace=["Search papers", "Retrieve evidence", "Synthesize answer"],
    )

    assert result.question == "What is the proposed methodology?"
    assert len(result.citations) == 1
    assert result.citations[0].source_id == "S1"
    assert len(result.sources) == 1
    assert len(result.research_trace) == 3

    json_str = result.model_dump_json()
    result_reconstructed = ResearchResult.model_validate_json(json_str)

    assert result_reconstructed == result
