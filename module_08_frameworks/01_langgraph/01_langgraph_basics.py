"""
LangGraph Basics — The Smallest Possible Graph
================================================
This shows the core mechanics: State, Nodes, Edges.
No LLM needed — just pure Python to see how data flows.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add


# ============================================================
# STEP 1: Define STATE — the data structure that flows through
# ============================================================
# Think of this as the "messages" list from your ReAct agent,
# but now it can hold ANY structured data.

class AgentState(TypedDict):
    # 'messages' accumulates (reducer=add means append, not overwrite)
    messages: Annotated[list[str], add]
    # 'step_count' tracks how many nodes we've visited
    step_count: int


# ============================================================
# STEP 2: Define NODES — functions that transform state
# ============================================================
# Each node takes state, does work, returns PARTIAL state update.
# Only the keys you return get updated. Other keys stay unchanged.

def greet(state: AgentState) -> dict:
    """First node: adds a greeting message."""
    print(f"  [greet node] Current messages: {state['messages']}")
    print(f"  [greet node] Current step_count: {state['step_count']}")
    return {
        "messages": ["Hello! I'm your research agent."],
        "step_count": state["step_count"] + 1
    }


def research(state: AgentState) -> dict:
    """Second node: simulates doing research."""
    print(f"  [research node] Current messages: {state['messages']}")
    print(f"  [research node] Current step_count: {state['step_count']}")
    return {
        "messages": ["I found 3 papers on LangGraph."],
        "step_count": state["step_count"] + 1
    }


def summarize(state: AgentState) -> dict:
    """Third node: summarizes the research."""
    print(f"  [summarize node] Current messages: {state['messages']}")
    print(f"  [summarize node] Current step_count: {state['step_count']}")
    return {
        "messages": [f"Summary: Visited {state['step_count'] + 1} nodes total."],
        "step_count": state["step_count"] + 1
    }


# ============================================================
# STEP 3: Build the GRAPH — connect nodes with edges
# ============================================================

# Create graph with our state type
graph_builder = StateGraph(AgentState)

# Add nodes (name → function)
graph_builder.add_node("greet", greet)
graph_builder.add_node("research", research)
graph_builder.add_node("summarize", summarize)

# Add edges (connections between nodes)
graph_builder.add_edge(START, "greet")        # Entry point
graph_builder.add_edge("greet", "research")   # greet → research
graph_builder.add_edge("research", "summarize")  # research → summarize
graph_builder.add_edge("summarize", END)      # summarize → exit

# Compile the graph (validates it, makes it runnable)
graph = graph_builder.compile()


# ============================================================
# STEP 4: RUN IT — watch the data flow!
# ============================================================

print("=" * 60)
print("LANGGRAPH BASICS — Watching Data Flow Through a Graph")
print("=" * 60)

# Initial state
initial_state = {
    "messages": ["User: What is LangGraph?"],
    "step_count": 0
}

print(f"\n📥 INITIAL STATE:")
print(f"   messages:   {initial_state['messages']}")
print(f"   step_count: {initial_state['step_count']}")
print()

# Run the graph
print("🔄 EXECUTING GRAPH...")
print("-" * 60)
result = graph.invoke(initial_state)
print("-" * 60)

print(f"\n📤 FINAL STATE:")
print(f"   messages:   {result['messages']}")
print(f"   step_count: {result['step_count']}")

print(f"\n💡 KEY OBSERVATIONS:")
print(f"   1. 'messages' ACCUMULATED (reducer=add): started with 1, ended with {len(result['messages'])}")
print(f"   2. 'step_count' was OVERWRITTEN each time (no reducer): final = {result['step_count']}")
print(f"   3. Each node only returned PARTIAL updates — LangGraph merged them")
print(f"   4. The graph ran greet → research → summarize → END automatically")
