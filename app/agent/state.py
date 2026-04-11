from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Slide navigation
    current_slide: int                    # 0-based index of active slide
    target_slide: Optional[int]           # set by understand_node, consumed by navigate_node

    # Conversation
    messages: Annotated[list, add_messages]  # full chat history (LangGraph managed)
    transcript: str                          # latest user utterance

    # Agent output
    response_text: str                    # text the agent will speak
    slide_changed: bool                   # whether a slide change just occurred

    # Control flow
    interrupted: bool                     # set True when client sends interrupt
    should_navigate: bool                 # set by understand_node
