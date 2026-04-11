"""
LangGraph state machine for the voice slide agent.

Graph topology:
  START → understand → [conditional: should_navigate?]
                           ↓ yes          ↓ no
                       navigate        respond
                           ↓              ↓
                         respond ────────→ END

TTS runs outside the graph (WebSocket layer, Plan 04).
"""
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes import understand_node, navigate_node, respond_node, should_navigate


def build_graph():
    """Build and compile the agent graph. Call once at startup."""
    builder = StateGraph(AgentState)

    builder.add_node("understand", understand_node)
    builder.add_node("navigate", navigate_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        should_navigate,
        {"navigate": "navigate", "respond": "respond"},
    )
    builder.add_edge("navigate", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


# Module-level compiled graph — import this in the WebSocket handler
agent_graph = build_graph()
