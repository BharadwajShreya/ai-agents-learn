"""
LangGraph + Real LLM + Tools — Research Agent (via OpenRouter)
===============================================================
This builds a REAL agent using:
  - OpenRouter free model (Gemma 4 / Nemotron) as the LLM
  - Custom tools (calculator + web_search simulator)
  - LangGraph graph construction

KEY LEARNING: OpenRouter provides an OpenAI-compatible API, so we 
use LangChain's ChatOpenAI with a custom base_url. This pattern 
works for ANY OpenAI-compatible provider (Together AI, Groq, etc.)
"""

import os
from dotenv import load_dotenv

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# STEP 1: Define TOOLS
# ============================================================
# The @tool decorator + docstring = tool schema the LLM sees

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.
    Input should be a valid Python math expression like '2 + 3 * 4' or '(10 / 2) ** 3'.
    """
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this when you need facts or data.
    Input should be a search query string.
    """
    # Simulated results for demo
    results = {
        "langgraph": "LangGraph is a library by LangChain for building stateful, multi-actor applications with LLMs using graph-based workflows. It supports cycles, controllability, and persistence. Latest version: 0.6.x (2026).",
        "langchain": "LangChain is a framework for developing applications powered by LLMs. It provides abstractions for chains, agents, memory, and tool integration.",
        "india": "India's estimated population in 2026 is approximately 1.46 billion people, making it the most populous country in the world.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return value
    return f"Search results for '{query}': Found several relevant articles discussing {query}."


tools = [calculator, web_search]


# ============================================================
# STEP 2: Set up the LLM via OpenRouter
# ============================================================
# OpenRouter uses the OpenAI-compatible API format.
# We just point ChatOpenAI to OpenRouter's base URL.
#
# This pattern works for ANY OpenAI-compatible provider:
#   - OpenRouter: https://openrouter.ai/api/v1
#   - Together AI: https://api.together.xyz/v1
#   - Groq:        https://api.groq.com/openai/v1
#   - Ollama:      http://localhost:11434/v1

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",    # Free model with tool support
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",     # Required by OpenRouter
        "X-Title": "LangGraph Learning"         # Optional: shows in OpenRouter dashboard
    }
)

# bind_tools → LLM now knows about our tools and can call them
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# STEP 3: Define STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# STEP 4: Define NODES
# ============================================================

def agent_node(state: AgentState) -> dict:
    """The 'agent' node: calls the LLM with current message history."""
    print(f"\n  🤖 [agent node] Calling LLM with {len(state['messages'])} messages...")
    print("state[messages]",state["messages"])
    
    response = llm_with_tools.invoke(state["messages"])
    print("response",response)
    
    if response.tool_calls:
        print(f"  🤖 [agent node] LLM wants to call {len(response.tool_calls)} tool(s):")
        for tc in response.tool_calls:
            print(f"      → {tc['name']}({tc['args']})")
    else:
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"  🤖 [agent node] LLM returned final answer:")
        print(f"      → {content[:120]}...")
    
    return {"messages": [response]}


# Prebuilt ToolNode handles execution automatically
tool_node = ToolNode(tools)


# ============================================================
# STEP 5: Routing function (conditional edge)
# ============================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if the LLM wants to call tools or is done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ============================================================
# STEP 6: BUILD THE GRAPH
# ============================================================

print("=" * 60)
print("LANGGRAPH AGENT — Real LLM via OpenRouter")
print("=" * 60)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent", should_continue,
    {"tools": "tools", "end": END}
)
graph_builder.add_edge("tools", "agent")

agent = graph_builder.compile()

print("\n✅ Graph compiled!")
print("   Model: google/gemma-4-26b-a4b-it:free (via OpenRouter)")
print("   Tools: calculator, web_search")
print("   Graph: START → agent ⇄ tools → END")


# ============================================================
# STEP 7: RUN THE AGENT
# ============================================================

def run_agent(query: str):
    """Run the agent and show full trace."""
    print("\n" + "=" * 60)
    print(f"📝 QUERY: {query}")
    print("=" * 60)
    
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful research assistant. Use tools when needed. Be concise."),
            HumanMessage(content=query)
        ]
    }
    
    result = agent.invoke(initial_state)
    
    print("\n" + "-" * 60)
    print("📋 FULL MESSAGE TRACE:")
    print("-" * 60)
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  [{i}] {msg_type}: (tool_calls: {[tc['name'] for tc in msg.tool_calls]})")
        elif msg_type == "ToolMessage":
            print(f"  [{i}] {msg_type} ({msg.name}): {msg.content[:80]}")
        elif msg_type == "SystemMessage":
            print(f"  [{i}] {msg_type}: {msg.content[:60]}...")
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"  [{i}] {msg_type}: {content[:100]}")
    
    final_msg = result["messages"][-1]
    final_content = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
    print(f"\n🎯 FINAL ANSWER: {final_content[:200]}")
    return result


# --- Test Queries ---
print("\n" + "🔬 " * 20)
print("TEST 1: Math question (should use calculator)")
print("🔬 " * 20)
run_agent("What is 25 * 47 + 123?")

print("\n\n" + "🔬 " * 20)
print("TEST 2: Research question (should use web_search)")
print("🔬 " * 20)
run_agent("What is the current population of India?")

print("\n\n" + "🔬 " * 20)
print("TEST 3: Multi-step (should use both tools)")
print("🔬 " * 20)
run_agent("Search for India's population and then calculate what 15% of that number would be.")
