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


def build_routing_graph():
    """
    Routing-only graph: understand + navigate, no respond node.

    Used by the WebSocket handler for streaming generation — the handler runs
    intent parsing and navigation here, then streams the LLM response directly
    (sentence by sentence → TTS) without going back into the graph.

    Keeping agent_graph intact means all 77 existing tests remain unaffected.
    """
    builder = StateGraph(AgentState)

    builder.add_node("understand", understand_node)
    builder.add_node("navigate", navigate_node)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        should_navigate,
        {"navigate": "navigate", "respond": END},
    )
    builder.add_edge("navigate", END)

    return builder.compile()


# Module-level compiled graphs — import these in the WebSocket handler
agent_graph = build_graph()
routing_graph = build_routing_graph()
