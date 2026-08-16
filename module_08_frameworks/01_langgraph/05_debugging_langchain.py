"""
DEBUGGING LangGraph/LangChain — See the ACTUAL Prompts & Responses
===================================================================
Problem: LangChain wraps everything in message objects. You can't easily see:
  1. What ACTUAL JSON gets sent to the LLM API
  2. What ACTUAL JSON comes back
  3. How your tools get converted to function schemas

This script shows 4 debugging techniques:
  1. Manual message inspection (what you tried)
  2. LangChain callbacks (the proper way)
  3. LangChain debug mode (the nuclear option)
  4. Direct API comparison (see what LangChain hides)
"""

import os
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal, Any


# ============================================================
# Tools (same as before)
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
        "india": "India's estimated population in 2026 is approximately 1.46 billion.",
        "langgraph": "LangGraph is a graph-based framework for building agent workflows.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': Found relevant articles."

tools = [calculator, web_search]


# ============================================================
# TECHNIQUE 1: Inspect LangChain message objects
# ============================================================
# This is what you did — printing state["messages"] and response.
# The problem: you see LangChain's WRAPPER, not the raw API payload.

print("=" * 70)
print("TECHNIQUE 1: Understanding LangChain Message Objects")
print("=" * 70)

# Let's create messages and inspect what LangChain stores:
human_msg = HumanMessage(content="What is 5 + 3?")
system_msg = SystemMessage(content="You are a helpful assistant.")

print("\n📦 HumanMessage object:")
print(f"   .content     = {human_msg.content!r}")
print(f"   .type         = {human_msg.type!r}")
print(f"   .id           = {human_msg.id!r}")
print(f"   Full dict     = {human_msg.model_dump()}")

print("\n📦 SystemMessage object:")
print(f"   .type         = {system_msg.type!r}")

# What LangChain CONVERTS this to for the API:
print("\n🔄 What LangChain sends to the API (OpenAI format):")
print("   HumanMessage  → {'role': 'user', 'content': 'What is 5 + 3?'}")
print("   SystemMessage → {'role': 'system', 'content': 'You are a helpful assistant.'}")
print("   AIMessage     → {'role': 'assistant', 'content': '...', 'tool_calls': [...]}")
print("   ToolMessage   → {'role': 'tool', 'content': '...', 'tool_call_id': '...'}")


# ============================================================
# TECHNIQUE 2: See EXACT tool schemas sent to the LLM
# ============================================================
print("\n\n" + "=" * 70)
print("TECHNIQUE 2: See Tool Schemas (what the LLM actually sees)")
print("=" * 70)

# When you call bind_tools(), LangChain converts your @tool functions
# into JSON schemas. Let's see exactly what gets sent:

llm = ChatOpenAI(
    # model="google/gemma-4-26b-a4b-it:free",
    model="openai/gpt-oss-20b:free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangGraph Debug Demo"
    }
)

llm_with_tools = llm.bind_tools(tools)

# Extract the tool schemas from the bound LLM
print("\n🔧 Tool schemas sent to the LLM with every request:")
print("   (This is what bind_tools() generates from your @tool functions)")
print()

# The bound kwargs contain the tools parameter
bound_kwargs = llm_with_tools.kwargs
if "tools" in bound_kwargs:
    for i, tool_schema in enumerate(bound_kwargs["tools"]):
        print(f"   Tool {i+1}:")
        print(f"   {json.dumps(tool_schema, indent=6)}")
        print()

print("💡 KEY INSIGHT: Your @tool docstring becomes the 'description' field.")
print("   The LLM reads these descriptions to decide WHICH tool to call.")
print("   Bad docstrings = LLM picks wrong tools!")


# ============================================================
# TECHNIQUE 3: Custom Callback Handler (THE PROPER WAY)
# ============================================================
print("\n\n" + "=" * 70)
print("TECHNIQUE 3: Custom Callback Handler (see everything!)")
print("=" * 70)

class DebugCallbackHandler(BaseCallbackHandler):
    """
    LangChain fires callbacks at each step of the LLM call.
    This handler intercepts and prints the ACTUAL payloads.
    """
    
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        """Called when LLM is about to be invoked."""
        print("\n  ┌─── 📤 LLM REQUEST ───────────────────────────────")
        print(f"  │ Model: {serialized.get('id', ['unknown'])}")
        # For chat models, the actual messages are in kwargs
        if "messages" in kwargs.get("invocation_params", {}):
            msgs = kwargs["invocation_params"]["messages"]
            print(f"  │ Messages ({len(msgs)}):")
            for m in msgs:
                role = m.get("role", "?")
                content = m.get("content", "")
                if content:
                    print(f"  │   [{role}]: {str(content)[:80]}")
                if m.get("tool_calls"):
                    print(f"  │   [{role}]: tool_calls={m['tool_calls']}")
        print("  └──────────────────────────────────────────────────")
    
    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs):
        """Called specifically for chat models — gives us the actual messages."""
        print("\n  ┌─── 📤 CHAT MODEL REQUEST ─────────────────────────")
        print(f"  │ Model: {kwargs.get('invocation_params', {}).get('model', 'unknown')}")
        
        # messages is a list of lists (batches), we usually have 1 batch
        for batch_idx, batch in enumerate(messages):
            print(f"  │")
            print(f"  │ === ACTUAL MESSAGES SENT TO API ===")
            for msg in batch:
                msg_type = type(msg).__name__
                content = msg.content if hasattr(msg, 'content') else str(msg)
                content_preview = str(content)[:100] if content else "(empty)"
                
                if msg_type == "AIMessage" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  │   [{msg_type}]: tool_calls={[tc['name'] for tc in msg.tool_calls]}")
                elif msg_type == "ToolMessage":
                    tool_name = msg.name if hasattr(msg, 'name') else '?'
                    print(f"  │   [{msg_type}] ({tool_name}): {content_preview}")
                else:
                    print(f"  │   [{msg_type}]: {content_preview}")
        
        # Show the tools being sent
        tools_param = kwargs.get('invocation_params', {}).get('tools', [])
        if tools_param:
            tool_names = [t.get('function', {}).get('name', '?') for t in tools_param]
            print(f"  │")
            print(f"  │ Tools included: {tool_names}")
        
        print("  └──────────────────────────────────────────────────")
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM returns a response."""
        print("\n  ┌─── 📥 LLM RESPONSE ──────────────────────────────")
        
        for gen in response.generations:
            for g in gen:
                msg = g.message if hasattr(g, 'message') else g
                
                # Content
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if content:
                    print(f"  │ Content: {str(content)[:120]}")
                
                # Tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  │ Tool calls:")
                    for tc in msg.tool_calls:
                        print(f"  │   → {tc['name']}({json.dumps(tc['args'])})")
                
                # Token usage
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    usage = msg.usage_metadata
                    print(f"  │ Tokens: input={usage.get('input_tokens', '?')}, "
                          f"output={usage.get('output_tokens', '?')}, "
                          f"total={usage.get('total_tokens', '?')}")
                
                # Finish reason
                meta = msg.response_metadata if hasattr(msg, 'response_metadata') else {}
                if meta.get('finish_reason'):
                    print(f"  │ Finish reason: {meta['finish_reason']}")
                    
                # Cost (OpenRouter provides this!)
                token_usage = meta.get('token_usage', {})
                cost = token_usage.get('cost', None)
                if cost is not None:
                    print(f"  │ Cost: ${cost}")
        
        print("  └──────────────────────────────────────────────────")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Called when a tool is about to execute."""
        tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        print(f"\n  ⚡ TOOL EXECUTING: {tool_name}({input_str})")
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool returns."""
        print(f"  ⚡ TOOL RESULT: {str(output)[:100]}")


# ============================================================
# Build the agent with the debug callback
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(
        state["messages"],
        config={"callbacks": [DebugCallbackHandler()]}
    )
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "agent")
agent = graph_builder.compile()


print("\n\n🚀 Running agent WITH debug callbacks...")
print("   Watch how every LLM call shows the exact messages and response!\n")

result = agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant. Use tools when needed. Be concise."),
        HumanMessage(content="What is 42 * 58?")
    ]
})

print(f"\n\n🎯 Final answer: {result['messages'][-1].content}")


# ============================================================
# TECHNIQUE 4: LangChain Debug Mode (nuclear option)
# ============================================================
print("\n\n" + "=" * 70)
print("TECHNIQUE 4: LangChain Debug Mode")
print("=" * 70)
print("""
To enable MAXIMUM verbosity, add these lines at the top of your script:

    import langchain
    langchain.debug = True      # Prints EVERYTHING
    # OR
    langchain.verbose = True    # Prints a summary

This will show:
  - Every chain/node entry and exit
  - Full prompt sent to the LLM
  - Full response from the LLM
  - Tool inputs and outputs
  - State before and after each node

⚠️  WARNING: This produces A LOT of output. Use callbacks (Technique 3)
    for targeted debugging. Use debug=True only when you're truly stuck.
""")


# ============================================================
# SUMMARY: The Abstraction Layers
# ============================================================
print("=" * 70)
print("📊 SUMMARY: What LangChain/LangGraph Does Under the Hood")
print("=" * 70)
print("""
YOUR CODE                    LANGCHAIN LAYER               ACTUAL API CALL
──────────                   ───────────────               ──────────────
                                                           POST https://openrouter.ai/api/v1/
                                                                /chat/completions
                                                           
SystemMessage("Be helpful")  → {"role": "system",         ─┐
                                 "content": "Be helpful"}   │
                                                            │
HumanMessage("What is 5+3?") → {"role": "user",            │  {
                                 "content": "5+3?"}         │    "model": "google/gemma-4...",
                                                            ├─►  "messages": [...],
@tool calculator             → {"type": "function",         │    "tools": [...],
  def calc(expr: str)           "function": {               │    "temperature": 0
    '''Evaluate math'''          "name": "calculator",      │  }
                                 "description": "Evaluate   │
                                   math",                   │
                                 "parameters": {            │
                                   "type": "object",       ─┘
                                   "properties": {
                                     "expression": {
                                       "type": "string"
                                     }
                                   }
                                 }
                               }}

API RESPONSE                  LANGCHAIN WRAPS AS
──────────────                ─────────────────
{"choices": [{                AIMessage(
  "message": {                  content="",
    "role": "assistant",        tool_calls=[{
    "tool_calls": [{              "name": "calculator",
      "function": {               "args": {"expression": "5+3"},
        "name": "calculator",     "id": "call_ABC123"
        "arguments": "..."      }],
      },                        response_metadata={...},
      "id": "call_ABC123"       usage_metadata={...}
    }]                        )
  },
  "finish_reason": "tool_calls"
}]}

💡 INTERVIEW INSIGHT:
"LangChain's main value for LLM calls is abstracting the provider-specific
API format. Whether you use OpenAI, Anthropic, Google, or OpenRouter, 
your code uses the same HumanMessage/AIMessage objects. LangChain handles
the conversion to each provider's API format."
""")

print("\n" + "=" * 70)
print("🔑 DEBUGGING CHEAT SHEET")
print("=" * 70)
print("""
┌──────────────────┬──────────────────────────────────────────────┐
│ What to debug     │ How                                        │
├──────────────────┼──────────────────────────────────────────────┤
│ Tool schemas      │ print(llm.bind_tools(tools).kwargs['tools'])│
│ Messages in/out   │ Custom callback (on_chat_model_start/end)  │
│ State at each     │ Print in node functions                     │
│   graph node      │                                            │
│ Everything        │ langchain.debug = True                     │
│ Token usage       │ response.usage_metadata                    │
│ Cost tracking     │ response.response_metadata['token_usage']  │
│ Production        │ LangSmith (langsmith.com) — full tracing   │
│   monitoring      │   export LANGCHAIN_TRACING_V2=true         │
└──────────────────┴──────────────────────────────────────────────┘
""")
