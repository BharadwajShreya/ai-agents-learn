# Module 8.1: LangGraph — Deep Dive Notes

## Table of Contents
1. [Why Frameworks?](#1-why-frameworks)
2. [LangGraph Core Concepts](#2-langgraph-core-concepts)
3. [State & Reducers](#3-state--reducers)
4. [Nodes — Functions That Transform State](#4-nodes--functions-that-transform-state)
5. [Edges — Normal vs Conditional](#5-edges--normal-vs-conditional)
6. [Building a Graph — Step by Step](#6-building-a-graph--step-by-step)
7. [LangGraph with Real LLM & Tools](#7-langgraph-with-real-llm--tools)
8. [Prebuilt: create_react_agent()](#8-prebuilt-create_react_agent)
9. [LangChain Under the Hood — Abstraction Layers](#9-langchain-under-the-hood--abstraction-layers)
10. [Debugging LangChain/LangGraph](#10-debugging-langchainlanggraph)
11. [OpenRouter & Provider Abstraction](#11-openrouter--provider-abstraction)
12. [Comparison Tables](#12-comparison-tables)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. Why Frameworks?

### THE PROBLEM
In Module 7, we built a working ReAct agent with a simple `while` loop. It works for basic cases, but real-world agents need:
- Human-in-the-loop approval before tool calls
- Parallel tool execution (call 3 APIs at once)
- Conditional branching (if search fails → try database)
- Persistent checkpoints (resume after a crash)
- Streaming partial results to the user
- Multi-agent handoffs (Agent A delegates to Agent B)

### THE NAIVE APPROACH
"Why not just add if/else statements to the while loop?"
→ Your simple loop becomes **spaghetti code** with deeply nested conditionals, manual state tracking, and no way to visualize the flow.

### THE SOLUTION
Frameworks like **LangGraph** model agents as **state machines (directed graphs)**, which makes complex flows:
- **Declarative**: You describe the shape of the flow, not imperative control flow
- **Visualizable**: The graph structure IS the documentation
- **Extensible**: Adding a new branch = adding a node + edge, not refactoring the loop
- **Production-ready**: Built-in persistence, streaming, human-in-the-loop

### INTERVIEW ANGLE
> "I'd use a framework like LangGraph when my agent needs complex control flow — conditional branching, human approval steps, parallel tool calls, or persistence. For a simple single-tool ReAct loop, hand-rolling is fine. But the moment you need cycles with conditional exits, checkpointing, or multi-agent coordination, a graph-based framework pays for itself."

**Code reference:** `01_langgraph_basics.py` (pure Python demo, no LLM)

---

## 2. LangGraph Core Concepts

LangGraph has **3 core primitives**: State, Nodes, and Edges.

```
    ┌────────────────────────────────────────────────────────┐
    │               LANGGRAPH CORE CONCEPTS                  │
    ├────────────────────────────────────────────────────────┤
    │                                                        │
    │  STATE = A TypedDict that flows through the graph      │
    │          (like the "messages" list in your ReAct agent) │
    │                                                        │
    │  NODE  = A Python function that reads state,           │
    │          does work, and returns updated state           │
    │          (like "call_llm" or "run_tool")               │
    │                                                        │
    │  EDGE  = Connection between nodes                      │
    │          - Normal edge:      A ──→ B (always)          │
    │          - Conditional edge:  A ──→ B or C (based on   │
    │                               a routing function)      │
    │                                                        │
    │  START/END = Special nodes marking entry and exit      │
    │                                                        │
    └────────────────────────────────────────────────────────┘
```

### How the ReAct Pattern Maps to LangGraph

```
  HAND-ROLLED AGENT (Module 7):      LANGGRAPH EQUIVALENT:

  messages = []                       State = {"messages": []}
  
  response = llm(messages)            Node: "agent" (calls LLM)
  
  if "FINAL" in response:             Conditional Edge:
      return answer                     → END if done
  else:                                 → "tools" if action found
      run_tool(...)           
                                      Node: "tools" (runs tool)
  messages.append(result)            
  # loop back                         Edge: "tools" → "agent"

  ┌───────┐    has tool call?    ┌────────┐
  │ agent ├──── YES ────────────►│ tools  │
  │ (LLM) │                     │ (exec) │
  │       │◄────────────────────┤        │
  └───┬───┘     always          └────────┘
      │
      │ NO (final answer)
      ▼
   [ END ]
```

---

## 3. State & Reducers

### State Definition

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list[str], add]   # With reducer → ACCUMULATES
    step_count: int                        # Without reducer → OVERWRITES
```

- State is a **TypedDict** — a structured dictionary with typed fields
- Every node receives the full state and returns a **partial update** (only keys it changed)
- LangGraph **merges** the partial update into the existing state

### Reducers — The Merge Strategy

| State Field | Has Reducer? | Behavior | Example |
|-------------|-------------|----------|---------|
| `messages: Annotated[list, add]` | ✅ Yes (`add`) | **Accumulates** — appends new values to existing list | `["hi"] + ["hello"] = ["hi", "hello"]` |
| `step_count: int` | ❌ No | **Overwrites** — last write wins | `1 → 2 → 3` (only `3` is kept) |

### The `add_messages` Reducer (Production Standard)

For real agents, LangGraph provides `add_messages` — smarter than plain `add`:

```python
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` is smarter because it:
- Handles **message deduplication** by ID
- Properly matches **ToolMessages to their AIMessage tool_calls**
- This is the standard reducer for all LangGraph agents

### Worked Example — State Flow

```
Initial:     messages = ["User: hi"],     step_count = 0
                    │
  greet node returns: messages=["Hello!"], step_count=1
                    │
  After merge:  messages = ["User: hi", "Hello!"],  step_count = 1
                    │                                     │
                    └── ACCUMULATED (reducer=add)         └── OVERWRITTEN
```

**Code reference:** `01_langgraph_basics.py` — run and watch accumulation vs overwrite behavior

---

## 4. Nodes — Functions That Transform State

```python
def agent_think(state: AgentState) -> dict:
    """A node is just a Python function."""
    # 1. Read current state
    messages = state["messages"]
    # 2. Do work (call LLM, run tool, etc.)
    response = call_llm(messages)
    # 3. Return PARTIAL state update
    return {"messages": [response], "step_count": state["step_count"] + 1}
```

**Key rules:**
- **Input:** receives the **full state** (read any field)
- **Output:** returns a **partial update** (only the keys that changed)
- Keys you DON'T return stay unchanged
- The node function name becomes the node's ID in the graph

### Prebuilt ToolNode

LangGraph provides `ToolNode` — a prebuilt node that:
1. Reads `tool_calls` from the last `AIMessage`
2. Finds the matching tool function
3. Calls it with the provided args
4. Returns `ToolMessage` with results

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)  # Pass your @tool-decorated functions
```

This replaces the manual tool execution logic from Module 7.

---

## 5. Edges — Normal vs Conditional

### Normal Edge (Unconditional)
```python
graph.add_edge("tools", "agent")  # tools ALWAYS goes back to agent
```
- Deterministic: A always leads to B
- Used for fixed flow (e.g., after running a tool, always return to the agent)

### Conditional Edge
```python
def should_continue(state) -> Literal["tools", "end"]:
    """Routing function that inspects state and decides next node."""
    if state["current_step"] == "done":
        return "end"
    return "tools"

graph.add_conditional_edges(
    "agent",                # Source node
    should_continue,        # Routing function
    {"tools": "tools", "end": END}  # Map return values → destinations
)
```

- A **routing function** inspects the state and returns a string
- That string is mapped to the next node
- This replaces `if "FINAL ANSWER" in response` from the hand-rolled agent

**Code reference:** `02_conditional_edges.py` — simulated ReAct loop as a graph

---

## 6. Building a Graph — Step by Step

```python
from langgraph.graph import StateGraph, START, END

# 1. Create graph builder with state type
graph_builder = StateGraph(AgentState)

# 2. Add nodes (name → function)
graph_builder.add_node("agent", agent_think)
graph_builder.add_node("tools", tool_execute)

# 3. Add edges
graph_builder.add_edge(START, "agent")          # Entry point
graph_builder.add_conditional_edges(             # Conditional routing
    "agent", should_continue, 
    {"tools": "tools", "end": END}
)
graph_builder.add_edge("tools", "agent")        # Always loop back

# 4. Compile (validates graph + makes it runnable)
graph = graph_builder.compile()

# 5. Run
result = graph.invoke(initial_state)
```

---

## 7. LangGraph with Real LLM & Tools

### Tool Definition with @tool

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: valid Python math expression."""
    result = eval(expression, {"__builtins__": {}}, {})
    return f"Result: {expression} = {result}"
```

**Key insight:** The **docstring** becomes the tool description that the LLM reads to decide which tool to call. Bad docstrings = LLM picks wrong tools!

### bind_tools() — Teaching the LLM About Tools

```python
llm_with_tools = llm.bind_tools(tools)
```

This takes your `@tool` decorated functions and:
1. Extracts their **name**, **description** (docstring), and **parameter schema**
2. Converts them to the OpenAI function calling format (JSON schema)
3. Sends this schema with every LLM call

### Tool Schema — What bind_tools() Actually Generates

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluate a mathematical expression. Input: valid Python math expression.",
    "parameters": {
      "properties": {
        "expression": {
          "type": "string"
        }
      },
      "required": ["expression"],
      "type": "object"
    }
  }
}
```

**To inspect tool schemas at any time:**
```python
print(json.dumps(llm_with_tools.kwargs['tools'], indent=2))
```

### Complete Agent Flow (Traced Output)

Actual output from our agent when asked "What is 25 * 47 + 123?":

```
Messages:  [System] → [Human] → [AI+tool_call] → [Tool result] → [AI final]
Nodes:      START   →  agent  →    tools        →    agent      →   END

Message Trace:
  [0] SystemMessage: You are a helpful research assistant...
  [1] HumanMessage: What is 25 * 47 + 123?
  [2] AIMessage: (tool_calls: ['calculator'])
  [3] ToolMessage (calculator): Result: 25 * 47 + 123 = 1298
  [4] AIMessage: 25 * 47 + 123 = 1,298

🎯 FINAL ANSWER: 25 * 47 + 123 = 1,298
```

Multi-step trace (search + calculate):
```
  [0] SystemMessage: You are a helpful research assistant...
  [1] HumanMessage: Search for India's population and calculate 15% of it
  [2] AIMessage: (tool_calls: ['web_search'])          ← first tool call
  [3] ToolMessage (web_search): India's population ≈ 1.46 billion
  [4] AIMessage: (tool_calls: ['calculator'])          ← second tool call
  [5] ToolMessage (calculator): 1460000000 * 0.15 = 219000000.0
  [6] AIMessage: India's population ≈ 1.46 billion, 15% = 219 million

  Graph path: agent → tools → agent → tools → agent → END
              (search)        (calculate)       (answer)
```

**Code reference:** `03_langgraph_real_agent.py`

---

## 8. Prebuilt: create_react_agent()

Everything built manually (State, agent node, tool node, conditional edges) can be done in **one line**:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful research assistant."
)

# Run it — same interface
result = agent.invoke({"messages": [HumanMessage(content="What is 5+3?")]})
```

### Manual Build vs Prebuilt Shortcut

```
Manual build (03_langgraph_real_agent.py):      Prebuilt shortcut:
  - Define AgentState TypedDict                  agent = create_react_agent(
  - Write agent_node function                        model=llm,
  - Create ToolNode                                  tools=tools
  - Write should_continue routing function       )
  - Build StateGraph, add nodes, edges
  - compile()
  = ~60 lines of code                           = 1 line of code ✨
```

Both produce the **identical graph**: `START → agent ⇄ tools → END`

### When to Use Each

```
┌─────────────────────────────────┬──────────────────────────────┐
│  create_react_agent()           │  Manual StateGraph           │
│  (the shortcut)                 │  (the custom build)          │
├─────────────────────────────────┼──────────────────────────────┤
│ ✅ Simple ReAct agent           │ ✅ Custom state fields        │
│ ✅ Standard tool calling        │ ✅ Custom routing logic       │
│ ✅ Quick prototyping            │ ✅ Human-in-the-loop         │
│ ✅ 90% of use cases             │ ✅ Multi-agent handoffs      │
│                                 │ ✅ Parallel branches         │
│ ❌ Can't add custom nodes       │ ✅ Custom nodes (validation, │
│ ❌ Can't change graph shape     │    summarization, etc.)      │
│ ❌ Limited routing control      │ ✅ Complex conditional flows  │
└─────────────────────────────────┴──────────────────────────────┘
```

### INTERVIEW ANGLE
> "I'd start with create_react_agent() for rapid prototyping, then switch to manual StateGraph when I need custom control flow, additional state fields, or human-in-the-loop checkpoints."

**Code reference:** `04_prebuilt_react.py`

---

## 9. LangChain Under the Hood — Abstraction Layers

### The 5 Layers of a LangGraph LLM Call

```
Layer 1: YOUR CODE          → message objects (HumanMessage, SystemMessage)
Layer 2: LANGCHAIN          → converts to OpenAI API format (JSON)
Layer 3: HTTP REQUEST       → POST to provider endpoint
Layer 4: LLM RESPONSE       → JSON from provider
Layer 5: LANGCHAIN          → wraps in AIMessage with metadata
```

### What LangChain Converts

```
YOUR CODE                        → ACTUAL API PAYLOAD
──────────                         ──────────────────

SystemMessage("Be helpful")      → {"role": "system", "content": "Be helpful"}
HumanMessage("What is 5+3?")    → {"role": "user", "content": "What is 5+3?"}
AIMessage(content="8")           → {"role": "assistant", "content": "8"}
AIMessage(tool_calls=[...])      → {"role": "assistant", "tool_calls": [...]}
ToolMessage(content="Result: 8") → {"role": "tool", "content": "Result: 8",
                                      "tool_call_id": "call_ABC123"}
```

### Full API Call Visualization

```
YOUR CODE                    LANGCHAIN LAYER               ACTUAL API CALL
──────────                   ───────────────               ──────────────
                                                           POST https://openrouter.ai/api/v1/
                                                                /chat/completions

SystemMessage("Be helpful")  → {"role": "system",         ─┐
                                 "content": "Be helpful"}   │
                                                            │
HumanMessage("5+3?")        → {"role": "user",             │  {
                                 "content": "5+3?"}         │    "model": "google/gemma-4...",
                                                            ├─►  "messages": [...],
@tool calculator             → {"type": "function",         │    "tools": [...],
  def calc(expr: str)           "function": {               │    "temperature": 0
    '''Evaluate math'''          "name": "calculator",      │  }
                                 "description": "...",      │
                                 "parameters": {...}       ─┘
                               }}
```

```
API RESPONSE                  LANGCHAIN WRAPS AS
──────────────                ─────────────────
{"choices": [{                AIMessage(
  "message": {                  content="",
    "role": "assistant",        tool_calls=[{
    "tool_calls": [{              "name": "calculator",
      "function": {               "args": {"expression": "5+3"},
        "name": "calculator",     "id": "call_ABC123"
        "arguments": "..."      }],
      },                        response_metadata={
      "id": "call_ABC123"         "token_usage": {...},
    }]                            "finish_reason": "tool_calls",
  },                              "model_name": "..."
  "finish_reason": "tool_calls"  },
}],                             usage_metadata={
"usage": {                        "input_tokens": 195,
  "prompt_tokens": 195,          "output_tokens": 23,
  "completion_tokens": 23        "total_tokens": 218
}}                              }
                              )
```

### INTERVIEW INSIGHT
> "LangChain's main value for LLM calls is abstracting the provider-specific API format. Whether you use OpenAI, Anthropic, Google, or OpenRouter, your code uses the same HumanMessage/AIMessage objects. LangChain handles the conversion to each provider's API format."

---

## 10. Debugging LangChain/LangGraph

### 4 Debugging Techniques

#### Technique 1: Inspect Message Objects
```python
# In your node function:
def agent_node(state):
    print("Messages:", state["messages"])
    response = llm_with_tools.invoke(state["messages"])
    print("Response:", response)
    print("Tool calls:", response.tool_calls)
    print("Content:", response.content)
    return {"messages": [response]}
```
**Limitation:** Shows LangChain wrappers, not the raw API payload.

#### Technique 2: Inspect Tool Schemas
```python
# See what bind_tools() generates:
llm_with_tools = llm.bind_tools(tools)
print(json.dumps(llm_with_tools.kwargs['tools'], indent=2))
```
**Use when:** LLM picks wrong tools or sends bad arguments.

#### Technique 3: Custom Callback Handler (RECOMMENDED)
```python
from langchain_core.callbacks import BaseCallbackHandler

class DebugCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        """See the EXACT messages going to the API."""
        print("📤 MESSAGES SENT:")
        for batch in messages:
            for msg in batch:
                print(f"  [{type(msg).__name__}]: {msg.content[:80]}")
        
        # See tools being sent
        tools = kwargs.get('invocation_params', {}).get('tools', [])
        if tools:
            print(f"  Tools: {[t['function']['name'] for t in tools]}")
    
    def on_llm_end(self, response, **kwargs):
        """See the EXACT response from the API."""
        for gen in response.generations:
            for g in gen:
                msg = g.message
                if msg.tool_calls:
                    print(f"📥 TOOL CALLS: {[tc['name'] for tc in msg.tool_calls]}")
                if msg.content:
                    print(f"📥 CONTENT: {msg.content[:100]}")
                print(f"📥 TOKENS: {msg.usage_metadata}")

# Use it:
response = llm_with_tools.invoke(
    messages,
    config={"callbacks": [DebugCallbackHandler()]}
)
```

#### Technique 4: LangChain Debug Mode (Nuclear Option)
```python
import langchain
langchain.debug = True      # Prints EVERYTHING
# OR
langchain.verbose = True    # Prints a summary
```
**⚠️ Warning:** Produces A LOT of output. Use callbacks for targeted debugging.

### Actual Debug Output

Running with the debug callback handler produces:
```
  ┌─── 📤 CHAT MODEL REQUEST ─────────────────────────
  │ Model: openai/gpt-oss-20b:free
  │
  │ === ACTUAL MESSAGES SENT TO API ===
  │   [SystemMessage]: You are a helpful assistant. Use tools when needed. Be concise.
  │   [HumanMessage]: What is 42 * 58?
  │
  │ Tools included: ['calculator', 'web_search']
  └──────────────────────────────────────────────────

  ┌─── 📥 LLM RESPONSE ──────────────────────────────
  │ Content: 42 × 58 = 2436
  │ Tokens: input=176, output=86, total=262
  │ Finish reason: stop
  │ Cost: $0
  └──────────────────────────────────────────────────
```

### Debugging Cheat Sheet

| What to Debug | How |
|--------------|-----|
| Tool schemas | `print(llm.bind_tools(tools).kwargs['tools'])` |
| Messages in/out | Custom callback (`on_chat_model_start/end`) |
| State at each graph node | `print()` in node functions |
| Everything | `langchain.debug = True` |
| Token usage | `response.usage_metadata` |
| Cost tracking | `response.response_metadata['token_usage']['cost']` |
| Production monitoring | LangSmith (`export LANGCHAIN_TRACING_V2=true`) |

**Code reference:** `05_debugging_langchain.py`

---

## 11. OpenRouter & Provider Abstraction

### Using OpenRouter with LangChain

OpenRouter provides an **OpenAI-compatible API**, so we use `ChatOpenAI` with a custom `base_url`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "My Agent"
    }
)
```

### This Pattern Works for ANY OpenAI-Compatible Provider

```python
# OpenRouter
openai_api_base = "https://openrouter.ai/api/v1"

# Together AI
openai_api_base = "https://api.together.xyz/v1"

# Groq
openai_api_base = "https://api.groq.com/openai/v1"

# Ollama (local)
openai_api_base = "http://localhost:11434/v1"
```

### Free Models with Tool Support (via OpenRouter)

| Model | Context | Best For |
|-------|---------|----------|
| `google/gemma-4-26b-a4b-it:free` | 262K | General purpose, good tool calling |
| `nvidia/nemotron-3-super-120b-a12b:free` | 262K | Larger model, better reasoning |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 1M | Largest free model |
| `openai/gpt-oss-20b:free` | 131K | OpenAI's open model |

### INTERVIEW INSIGHT
> "The OpenAI API format has become the de facto standard. Most providers — OpenRouter, Together AI, Groq, Ollama — expose an OpenAI-compatible endpoint. This means you can swap LLM providers by changing just the model name and base URL — the rest of your LangGraph code stays identical."

---

## 12. Comparison Tables

### Hand-Rolled Agent (Module 7) vs LangGraph

| Aspect | Hand-Rolled (Module 7) | LangGraph |
|--------|----------------------|-----------|
| Control flow | `while True` + `if/else` | Graph with nodes + edges |
| State | `messages = []` (just a list) | `TypedDict` with typed fields + reducers |
| Branching | Nested `if` statements | Conditional edges with routing functions |
| Adding a new step | Modify the while loop body | Add a node + edge |
| Persistence | Manual (save to file) | Built-in checkpointing |
| Visualization | Read the code | Graph structure IS the diagram |
| Human-in-the-loop | Manual `input()` calls | Built-in interrupt mechanism |
| Streaming | Manual print statements | Built-in streaming support |
| Multi-agent | Manual orchestration | Built-in subgraph support |

### LangChain Message Types

| LangChain Type | API Role | Purpose |
|---------------|----------|---------|
| `SystemMessage` | `system` | Instructions for the LLM |
| `HumanMessage` | `user` | User's input |
| `AIMessage` | `assistant` | LLM's response (text or tool_calls) |
| `ToolMessage` | `tool` | Result of a tool execution |

---

## 13. Interview Questions & Answers

### Q1: What is LangGraph and how does it differ from LangChain?
> "LangChain is a framework for building LLM applications with abstractions for chains, tools, and memory. LangGraph is a library built on top of LangChain that specifically models agent workflows as directed graphs with State, Nodes, and Edges. The key difference is that LangGraph supports cycles (loops), which are essential for agentic patterns like ReAct where the agent iterates between thinking and acting until it has enough information."

### Q2: What are the three core primitives of LangGraph?
> "State, Nodes, and Edges. State is a TypedDict that flows through the graph — it holds all the data (messages, counters, flags). Nodes are Python functions that receive state, do work, and return partial state updates. Edges connect nodes — normal edges are unconditional (always go from A to B), and conditional edges use a routing function to decide which node to visit next based on the current state."

### Q3: What is a reducer in LangGraph state?
> "A reducer defines how state updates are merged. When a node returns a partial state update, LangGraph uses the reducer to combine it with the existing state. For example, `Annotated[list, add]` means 'append new items to the existing list' (accumulate), while a field without a reducer gets overwritten entirely. The standard `add_messages` reducer is smarter — it handles deduplication and properly sequences tool call responses."

### Q4: When would you use create_react_agent() vs building a custom StateGraph?
> "I'd start with create_react_agent() for rapid prototyping — it handles 90% of use cases in one line. I'd switch to manual StateGraph when I need custom state fields, complex routing logic, human-in-the-loop approval steps, parallel branches, or multi-agent coordination. The prebuilt agent is limited to the standard ReAct pattern; the manual approach gives full control over the graph shape."

### Q5: How do you debug what LangChain actually sends to the LLM?
> "Four techniques: (1) Inspect tool schemas with `llm.bind_tools(tools).kwargs['tools']` to see the JSON schemas generated from @tool functions. (2) Use a custom callback handler — override `on_chat_model_start` to see messages going out and `on_llm_end` to see responses coming back. (3) Set `langchain.debug = True` for maximum verbosity. (4) In production, use LangSmith for full distributed tracing."

### Q6: How does LangChain abstract LLM providers?
> "LangChain converts your code's HumanMessage/AIMessage objects to the provider-specific API format. For OpenAI-compatible APIs (OpenRouter, Groq, Together), it serializes to the OpenAI chat completions format. For Anthropic, it converts to the Messages API format. For Google, it uses the Gemini format. Your code stays the same — you just change the model class and credentials."

---

## Code Files Reference

| File | What It Demonstrates |
|------|---------------------|
| `01_langgraph_basics.py` | Core mechanics: State, Nodes, Edges with pure Python (no LLM) |
| `02_conditional_edges.py` | ReAct pattern as a graph with conditional edges (simulated) |
| `03_langgraph_real_agent.py` | Full LangGraph agent with real LLM (OpenRouter) and tools |
| `04_prebuilt_react.py` | `create_react_agent()` shortcut vs manual build |
| `05_debugging_langchain.py` | 4 debugging techniques: message inspection, tool schemas, callbacks, debug mode |
