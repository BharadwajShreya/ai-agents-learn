"""
LangGraph Conditional Edges — The ReAct Pattern as a Graph
===========================================================
This recreates YOUR Module 7 ReAct agent as a LangGraph graph,
but without an actual LLM — we simulate decisions to show the mechanics.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from operator import add


# ============================================================
# STATE: What data flows through the graph
# ============================================================
class ReActState(TypedDict):
    messages: Annotated[list[str], add]
    current_step: str     # "think", "act", or "done"
    iteration: int


# ============================================================
# NODES: Functions that process state
# ============================================================
def agent_think(state: ReActState) -> dict:
    """Simulates the LLM thinking step."""
    iteration = state["iteration"] + 1
    
    # Simulate: after 3 iterations, decide we're done
    if iteration >= 3:
        thought = f"Iteration {iteration}: I now have enough info. FINAL ANSWER: LangGraph is a framework for building agent workflows as graphs."
        return {
            "messages": [f"💭 THOUGHT: {thought}"],
            "current_step": "done",
            "iteration": iteration
        }
    else:
        thought = f"Iteration {iteration}: I need to search for more information about LangGraph."
        action = "search" if iteration == 1 else "lookup"
        return {
            "messages": [f"💭 THOUGHT: {thought}", f"🔧 ACTION: {action}('LangGraph features')"],
            "current_step": "act",
            "iteration": iteration
        }


def tool_execute(state: ReActState) -> dict:
    """Simulates running a tool."""
    results = {
        1: "📋 OBSERVATION: LangGraph supports stateful, multi-actor applications with cycles.",
        2: "📋 OBSERVATION: LangGraph provides built-in persistence and human-in-the-loop."
    }
    observation = results.get(state["iteration"], "📋 OBSERVATION: No more results.")
    return {
        "messages": [observation],
        "current_step": "think"
    }


# ============================================================
# ROUTING FUNCTION: Decides which edge to take
# ============================================================
def should_continue(state: ReActState) -> Literal["tools", "end"]:
    """
    This is the CONDITIONAL EDGE function.
    It looks at state and decides: go to tools, or end?
    
    This replaces your 'if "FINAL ANSWER" in response' check!
    """
    if state["current_step"] == "done":
        return "end"
    else:
        return "tools"


# ============================================================
# BUILD THE GRAPH
# ============================================================
graph_builder = StateGraph(ReActState)

# Add nodes
graph_builder.add_node("agent", agent_think)
graph_builder.add_node("tools", tool_execute)

# Add edges
graph_builder.add_edge(START, "agent")  # Start → agent

# CONDITIONAL EDGE: agent → tools OR agent → END
graph_builder.add_conditional_edges(
    "agent",                    # Source node
    should_continue,            # Routing function
    {                           # Map return values to destinations
        "tools": "tools",       # If function returns "tools" → go to tools node
        "end": END              # If function returns "end" → stop
    }
)

graph_builder.add_edge("tools", "agent")  # tools always → back to agent

# Compile
graph = graph_builder.compile()


# ============================================================
# RUN AND WATCH
# ============================================================
print("=" * 60)
print("REACT PATTERN IN LANGGRAPH — Conditional Edge Demo")
print("=" * 60)
print()
print("Graph structure:")
print("  ┌───────┐   'tools'   ┌────────┐")
print("  │ agent ├────────────►│ tools  │")
print("  │ (LLM) │             │ (exec) │")
print("  │       │◄────────────┤        │")
print("  └───┬───┘   always    └────────┘")
print("      │")
print("      │ 'end'")
print("      ▼")
print("   [ END ]")
print()

initial = {
    "messages": ["User: What is LangGraph?"],
    "current_step": "think",
    "iteration": 0
}

print("📥 Starting with:", initial["messages"][0])
print("-" * 60)

# Run the graph
result = graph.invoke(initial)

print("-" * 60)
print(f"\n📤 FINAL STATE:")
print(f"   Total iterations: {result['iteration']}")
print(f"   Final step: {result['current_step']}")
print(f"\n📝 Full message history:")
for i, msg in enumerate(result["messages"]):
    print(f"   [{i}] {msg}")

print(f"\n💡 KEY OBSERVATION:")
print(f"   The graph looped agent→tools→agent→tools→agent→END")
print(f"   The CONDITIONAL EDGE decided 'tools' vs 'end' each time.")
print(f"   This is EXACTLY your Module 7 ReAct loop, but as a graph!")
