"""
LangGraph Prebuilt — create_react_agent() Shortcut
====================================================
Everything we built manually in 03_langgraph_real_agent.py
(State, agent node, tool node, conditional edges) can be
done in ONE LINE with create_react_agent().

This shows:
1. The shortcut version
2. How it's identical to our manual graph
3. When to use each approach
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ============================================================
# Same tools as before
# ============================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for information. Input: search query string."""
    results = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor agent applications with LLMs using graph-based workflows.",
        "population india": "India's estimated population in 2026 is approximately 1.46 billion people.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': Found several relevant articles."


tools = [calculator, web_search]


# ============================================================
# LLM setup (same OpenRouter config)
# ============================================================

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangGraph Learning"
    }
)


# ============================================================
# THE SHORTCUT: create_react_agent()
# ============================================================
# 
# THIS ONE LINE replaces ALL of this from our manual build:
#   - AgentState TypedDict
#   - agent_node function  
#   - ToolNode setup
#   - should_continue routing function
#   - StateGraph construction
#   - add_node, add_edge, add_conditional_edges
#   - compile()
#
# It creates the EXACT SAME graph:
#   START → agent ⇄ tools → END

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful research assistant. Use tools when needed. Be concise."
)

print("=" * 60)
print("CREATE_REACT_AGENT — The Production Shortcut")
print("=" * 60)
print()
print("Manual build (03_langgraph_real_agent.py):")
print("  - Define AgentState TypedDict")
print("  - Write agent_node function")
print("  - Create ToolNode")
print("  - Write should_continue routing function")
print("  - Build StateGraph, add nodes, edges, compile")
print("  = ~60 lines of code")
print()
print("Prebuilt shortcut:")
print("  agent = create_react_agent(model=llm, tools=tools)")
print("  = 1 line of code ✨")
print()
print("Both produce the IDENTICAL graph:")
print("  START → agent ⇄ tools → END")


# ============================================================
# Run it — exact same interface as our manual agent
# ============================================================

def run_query(query: str):
    print(f"\n{'=' * 60}")
    print(f"📝 QUERY: {query}")
    print(f"{'=' * 60}")
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        print("\n📋 MESSAGE TRACE:")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"  [{i}] {msg_type}: (tool_calls: {[tc['name'] for tc in msg.tool_calls]})")
            elif msg_type == "ToolMessage":
                print(f"  [{i}] {msg_type} ({msg.name}): {msg.content[:80]}")
            else:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"  [{i}] {msg_type}: {content[:100]}")
        
        final = result["messages"][-1]
        final_content = final.content if isinstance(final.content, str) else str(final.content)
        print(f"\n🎯 ANSWER: {final_content[:200]}")
        
    except Exception as e:
        print(f"\n❌ Error (free model timeout/rate limit): {type(e).__name__}: {str(e)[:100]}")
        print("   This is normal for free models — retry or use a different model.")


# Test
run_query("What is 144 * 12 - 56?")
run_query("What is the population of India?")


# ============================================================
# WHEN TO USE EACH APPROACH
# ============================================================
print("\n\n" + "=" * 60)
print("📊 WHEN TO USE EACH APPROACH")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│  create_react_agent()        │  Manual StateGraph           │
│  (the shortcut)              │  (the custom build)          │
├──────────────────────────────┼──────────────────────────────┤
│ ✅ Simple ReAct agent        │ ✅ Custom state fields        │
│ ✅ Standard tool calling     │ ✅ Custom routing logic       │
│ ✅ Quick prototyping         │ ✅ Human-in-the-loop         │
│ ✅ 90% of use cases          │ ✅ Multi-agent handoffs      │
│                              │ ✅ Parallel branches         │
│ ❌ Can't add custom nodes    │ ✅ Custom nodes (validation, │
│ ❌ Can't change graph shape  │    summarization, etc.)      │
│ ❌ Limited routing control   │ ✅ Complex conditional flows  │
└──────────────────────────────┴──────────────────────────────┘

Interview answer: "I'd start with create_react_agent() for
rapid prototyping, then switch to manual StateGraph when I
need custom control flow, additional state fields, or
human-in-the-loop checkpoints."
""")
