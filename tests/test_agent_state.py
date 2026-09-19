"""Unit tests for ResearchPilot LangGraph agent state (Step 0.11).

Verifies:
- ResearchState typed dictionary schema and fields.
- create_initial_state factory helper.
- Ambiguity-free state creation and incremental updates.
- Full compatibility with LangGraph's StateGraph.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.agent.state import ResearchState, create_initial_state
from app.models.chunk import Evidence
from app.models.paper import ScoredPaper


def test_research_state_fields():
    """Verify that ResearchState has all required fields."""
    expected_fields = {
        "question",
        "messages",
        "candidate_papers",
        "evidence",
        "selected_sources",
        "answer",
        "research_trace",
    }
    assert set(ResearchState.__annotations__.keys()) == expected_fields


def test_create_initial_state_defaults():
    """Verify initial state construction with default messages."""
    q = "What is the architecture of VideoMAE?"
    state = create_initial_state(q)

    assert state["question"] == q
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == q
    assert state["candidate_papers"] == []
    assert state["evidence"] == []
    assert state["selected_sources"] == []
    assert state["answer"] == ""
    assert state["research_trace"] == []


def test_create_initial_state_custom_messages():
    """Verify initial state construction with explicit custom messages."""
    q = "How does SlowFast handle high frame rate?"
    custom_msgs = [HumanMessage(content=q), AIMessage(content="I will search for SlowFast.")]
    state = create_initial_state(q, messages=custom_msgs)

    assert state["question"] == q
    assert len(state["messages"]) == 2
    assert state["messages"][1].content == "I will search for SlowFast."


def test_state_updates_without_ambiguity():
    """Verify that ResearchState can be updated incrementally without ambiguity."""
    state = create_initial_state("Explain C3D.")

    paper = ScoredPaper(
        paper_id="c3d_paper",
        title="Learning Spatiotemporal Features with 3D Convolutional Networks",
        score=0.85,
    )
    evidence_item = Evidence(
        source_id="S1",
        chunk_id="c3d_c0001",
        paper_id="c3d_paper",
        page=2,
        section="Methodology",
        text="3D ConvNets are more suitable for spatiotemporal feature learning.",
        score=0.82,
    )

    # Simulate state updates across agent steps
    state["candidate_papers"].append(paper)
    state["evidence"].append(evidence_item)
    state["selected_sources"].append("S1")
    state["answer"] = "C3D uses 3x3x3 convolutions [S1]."
    state["research_trace"].extend(["Searched papers", "Retrieved evidence", "Synthesized answer"])

    assert len(state["candidate_papers"]) == 1
    assert state["candidate_papers"][0].paper_id == "c3d_paper"
    assert len(state["evidence"]) == 1
    assert state["evidence"][0].source_id == "S1"
    assert state["selected_sources"] == ["S1"]
    assert state["answer"] == "C3D uses 3x3x3 convolutions [S1]."
    assert state["research_trace"] == ["Searched papers", "Retrieved evidence", "Synthesized answer"]


def test_langgraph_compatibility():
    # Note: type: ignore is required due to a known Pyrefly type checker issue where
    # TypedDict is prematurely erased during subtyping checks against LangGraph's TypedDictLike protocol.
    builder = StateGraph(ResearchState)  # type: ignore

    def dummy_node(state: ResearchState) -> dict:
        return {
            "answer": "Answer from dummy node.",
            "research_trace": ["Dummy step completed"],
        }

    builder.add_node("dummy", dummy_node)
    builder.add_edge(START, "dummy")
    builder.add_edge("dummy", END)

    graph = builder.compile()
    initial = create_initial_state("Test question")
    result = graph.invoke(initial)

    assert result["question"] == "Test question"
    assert result["answer"] == "Answer from dummy node."
    assert result["research_trace"] == ["Dummy step completed"]
