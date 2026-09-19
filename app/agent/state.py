"""LangGraph state representation for ResearchPilot MVP 0 (Step 0.11).

Defines the central agent state tracking the research question, conversation/tool messages,
candidate discovered papers, retrieved evidence passages, selected source IDs,
final answer, and the high-level research trace.
"""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.models.chunk import Evidence
from app.models.paper import ScoredPaper


class ResearchState(TypedDict):
    """LangGraph state representation for the ResearchPilot research agent.

    Attributes:
        question: The user's input technical research question.
        messages: Conversation and tool execution messages, managed via LangGraph's add_messages reducer.
        candidate_papers: Papers discovered during research via search_papers().
        evidence: Evidence chunks retrieved from indexed papers via retrieve_evidence().
        selected_sources: Sequential source IDs (e.g. ['S1', 'S2']) cited in the final answer.
        answer: The final citation-grounded research answer synthesized by the agent.
        research_trace: High-level milestone trace of agent actions (e.g. ['Searching papers', 'Retrieved evidence']).
    """

    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    candidate_papers: list[ScoredPaper]
    evidence: list[Evidence]
    selected_sources: list[str]
    answer: str
    research_trace: list[str]


def create_initial_state(
    question: str,
    messages: Sequence[BaseMessage] | None = None,
) -> ResearchState:
    """Create a clean, initialized ResearchState for a given research question.

    Args:
        question: The user's research question.
        messages: Optional initial message sequence. Defaults to [HumanMessage(content=question)].

    Returns:
        An initialized ResearchState dict ready for LangGraph execution.
    """
    initial_messages = list(messages) if messages is not None else [HumanMessage(content=question)]
    return {
        "question": question,
        "messages": initial_messages,
        "candidate_papers": [],
        "evidence": [],
        "selected_sources": [],
        "answer": "",
        "research_trace": [],
    }
