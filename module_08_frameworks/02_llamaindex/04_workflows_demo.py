"""
Module 8.2 — Script 04: LlamaIndex Workflows API
=================================================
LlamaIndex Workflows are the MODERN way to build complex agent pipelines.
They use an EVENT-DRIVEN architecture (vs LangGraph's GRAPH-BASED approach).

KEY MENTAL MODEL:
  LangGraph:   Nodes + Edges → state machine → state flows through graph
  LlamaIndex:  Steps + Events → pub/sub → events trigger steps

Both can build the same systems, but the programming model feels different:
  - LangGraph: "I'm drawing a flowchart" (visual, declarative)
  - Workflows: "I'm writing event handlers" (Pythonic, imperative)

WHEN TO USE WORKFLOWS:
- Already invested in LlamaIndex for retrieval
- Prefer async Python patterns over graph DSLs
- Need tight integration with LlamaIndex's data abstractions
- Building RAG-centric agents (retrieval is the core logic)

Run: source ../venv/bin/activate && python 04_workflows_demo.py
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Event,
    step,
    Context,
)
from llama_index.llms.openai import OpenAI

# Configure LLM
llm = OpenAI(
    model="google/gemma-3-27b-it:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
)


# ============================================================================
# DEMO 1: Minimal Workflow — Hello World
# ============================================================================
# Every workflow starts with StartEvent and ends with StopEvent.
# Steps are async methods decorated with @step.

print("=" * 70)
print("DEMO 1: Minimal Workflow (Hello World)")
print("=" * 70)


class HelloWorkflow(Workflow):
    """Simplest possible workflow: one step, in → out."""

    @step
    async def say_hello(self, ev: StartEvent) -> StopEvent:
        # ev.get() retrieves data passed to workflow.run()
        name = ev.get("name", "World")
        return StopEvent(result=f"Hello, {name}! 👋")


async def demo_hello():
    w = HelloWorkflow(timeout=10, verbose=False)
    result = await w.run(name="Sourav")
    print(f"Result: {result}")
    print()

asyncio.run(demo_hello())

# KEY POINT: @step methods define WHAT runs.
# The Event type hints define WHEN it runs (which event triggers it).
# StartEvent = "run this at the beginning"
# StopEvent = "I'm done, here's the result"


# ============================================================================
# DEMO 2: Multi-Step Workflow with Custom Events
# ============================================================================
# Custom Events are how you connect steps together.
# Step A emits EventX → Step B consumes EventX → Step B emits EventY → ...

print("=" * 70)
print("DEMO 2: Multi-Step Workflow (Custom Events)")
print("=" * 70)


# Define custom events — these are the "edges" between steps
class ClassifyEvent(Event):
    """Emitted after classification — carries the query + its category."""
    query: str
    category: str


class AnswerEvent(Event):
    """Emitted after answering — carries the final answer."""
    answer: str


class QueryClassifierWorkflow(Workflow):
    """
    2-step workflow:
      1. Classify the query (HR, Engineering, Product, or General)
      2. Generate a tailored response based on the category
    
    This is a common pattern: ROUTE first, then PROCESS.
    It's like a conditional edge in LangGraph, but expressed as event routing.
    """

    @step
    async def classify_query(self, ev: StartEvent) -> ClassifyEvent:
        """Step 1: Classify the incoming query."""
        query = ev.get("query")
        print(f"  📥 Step 1: Classifying query: \"{query}\"")

        response = await llm.acomplete(
            f"Classify this query into exactly one category: "
            f"HR, Engineering, Product, or General.\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )
        category = response.text.strip()
        print(f"  🏷️  Category: {category}")

        # Emit a ClassifyEvent → this triggers the next step
        return ClassifyEvent(query=query, category=category)

    @step
    async def generate_response(self, ev: ClassifyEvent) -> StopEvent:
        """Step 2: Generate a response tailored to the category."""
        print(f"  📤 Step 2: Generating {ev.category} response...")

        # Tailor the system prompt based on category
        persona_map = {
            "HR": "You are an HR specialist. Be warm and reference specific policies.",
            "Engineering": "You are a senior engineer. Be precise and reference best practices.",
            "Product": "You are a product manager. Focus on features and roadmap.",
            "General": "You are a helpful assistant. Be concise and friendly.",
        }
        persona = persona_map.get(ev.category, persona_map["General"])

        response = await llm.acomplete(
            f"{persona}\n\n"
            f"Answer this query concisely (2-3 sentences max):\n{ev.query}"
        )

        return StopEvent(result=f"[{ev.category}] {response.text.strip()}")


async def demo_classifier():
    w = QueryClassifierWorkflow(timeout=30, verbose=False)

    queries = [
        "How many vacation days do I get?",
        "What's the SLA for code reviews?",
        "What features are on the product roadmap?",
    ]

    for query in queries:
        print(f"\n📝 Query: \"{query}\"")
        result = await w.run(query=query)
        print(f"🤖 Answer: {result}")
    print()

asyncio.run(demo_classifier())


# ============================================================================
# DEMO 3: Workflow with Branching (Conditional Routing)
# ============================================================================
# Branching = a step emits DIFFERENT events based on a condition.
# This is the Workflow equivalent of LangGraph's conditional edges.

print("=" * 70)
print("DEMO 3: Branching Workflow (Conditional Routing)")
print("=" * 70)


class SimpleQueryEvent(Event):
    """Route for simple queries — answer directly."""
    query: str


class ComplexQueryEvent(Event):
    """Route for complex queries — decompose first, then answer."""
    query: str


class SubAnswerEvent(Event):
    """Carries a sub-answer from decomposed queries."""
    sub_answers: list[str]
    original_query: str


class SmartRouterWorkflow(Workflow):
    """
    Routes queries by complexity:
      Simple query  → answer directly
      Complex query → decompose into sub-questions → answer each → synthesize
    
    This demonstrates BRANCHING — one step, two possible paths.
    """

    @step
    async def route_query(self, ev: StartEvent) -> SimpleQueryEvent | ComplexQueryEvent:
        """Decide if the query is simple or complex."""
        query = ev.get("query")
        print(f"  🔀 Router: Analyzing complexity...")

        response = await llm.acomplete(
            f"Is this query simple (can be answered in one sentence) or "
            f"complex (requires multiple pieces of information)?\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY 'simple' or 'complex'."
        )
        complexity = response.text.strip().lower()
        print(f"  🔀 Router decision: {complexity}")

        if "complex" in complexity:
            return ComplexQueryEvent(query=query)
        else:
            return SimpleQueryEvent(query=query)

    @step
    async def handle_simple(self, ev: SimpleQueryEvent) -> StopEvent:
        """Direct answer for simple queries."""
        print(f"  ⚡ Simple path: Answering directly...")
        response = await llm.acomplete(
            f"Answer this briefly in 1-2 sentences:\n{ev.query}"
        )
        return StopEvent(result=f"[Simple] {response.text.strip()}")

    @step
    async def handle_complex(self, ev: ComplexQueryEvent) -> SubAnswerEvent:
        """Decompose complex queries into sub-questions and answer each."""
        print(f"  🧩 Complex path: Decomposing query...")

        # Step 1: Decompose
        decompose_response = await llm.acomplete(
            f"Break this complex query into 2-3 simple sub-questions. "
            f"Return each sub-question on a new line, numbered.\n\n"
            f"Query: {ev.query}"
        )
        sub_questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in decompose_response.text.strip().split("\n")
            if q.strip() and len(q.strip()) > 5
        ][:3]  # Limit to 3 sub-questions

        print(f"  🧩 Sub-questions: {sub_questions}")

        # Step 2: Answer each sub-question
        sub_answers = []
        for sq in sub_questions:
            answer = await llm.acomplete(f"Answer briefly: {sq}")
            sub_answers.append(f"Q: {sq}\nA: {answer.text.strip()}")

        return SubAnswerEvent(sub_answers=sub_answers, original_query=ev.query)

    @step
    async def synthesize(self, ev: SubAnswerEvent) -> StopEvent:
        """Combine sub-answers into a final response."""
        print(f"  📊 Synthesizing final answer...")
        combined = "\n\n".join(ev.sub_answers)
        response = await llm.acomplete(
            f"Based on these sub-answers, provide a comprehensive answer "
            f"to the original question.\n\n"
            f"Original question: {ev.original_query}\n\n"
            f"Sub-answers:\n{combined}\n\n"
            f"Provide a unified, concise answer (3-4 sentences)."
        )
        return StopEvent(result=f"[Complex] {response.text.strip()}")


async def demo_branching():
    w = SmartRouterWorkflow(timeout=60, verbose=False)

    # Simple query
    q1 = "What is the capital of France?"
    print(f"\n📝 Query: \"{q1}\"")
    r1 = await w.run(query=q1)
    print(f"🤖 Answer: {r1}")

    # Complex query
    q2 = "Compare the pros and cons of RAG vs fine-tuning for enterprise AI applications."
    print(f"\n📝 Query: \"{q2}\"")
    r2 = await w.run(query=q2)
    print(f"🤖 Answer: {r2}")
    print()

asyncio.run(demo_branching())


# ============================================================================
# DEMO 4: Workflow with Context (Shared State)
# ============================================================================
# Context is the shared state that persists across steps.
# Similar to LangGraph's TypedDict state, but more flexible.

print("=" * 70)
print("DEMO 4: Workflow with Context (Shared State)")
print("=" * 70)


class ProcessedEvent(Event):
    """Carries processed text."""
    text: str


class ContextWorkflow(Workflow):
    """
    Demonstrates using Context to persist state across steps.
    Context is like LangGraph's state — but it's a key-value store
    rather than a typed dictionary.
    """

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> ProcessedEvent:
        """Process input and store metadata in context."""
        text = ev.get("text")

        # Store data in context — accessible by ALL subsequent steps
        await ctx.set("original_length", len(text))
        await ctx.set("step_count", 1)

        processed = text.upper()
        print(f"  Step 1: Processed text ({len(text)} → {len(processed)} chars)")
        return ProcessedEvent(text=processed)

    @step
    async def summarize(self, ctx: Context, ev: ProcessedEvent) -> StopEvent:
        """Use context data from previous step."""
        # Read data from context
        original_length = await ctx.get("original_length")
        step_count = await ctx.get("step_count")

        # Update context
        await ctx.set("step_count", step_count + 1)
        final_step_count = await ctx.get("step_count")

        result = (
            f"Processed text: {ev.text[:50]}...\n"
            f"Original length: {original_length}\n"
            f"Total steps completed: {final_step_count}"
        )
        print(f"  Step 2: Summarized (total steps: {final_step_count})")
        return StopEvent(result=result)


async def demo_context():
    w = ContextWorkflow(timeout=10, verbose=False)
    result = await w.run(text="LlamaIndex workflows are event-driven and powerful.")
    print(f"Result:\n{result}")
    print()

asyncio.run(demo_context())


# ============================================================================
# COMPARISON: LangGraph vs LlamaIndex Workflows
# ============================================================================
print("=" * 70)
print("COMPARISON: LangGraph vs LlamaIndex Workflows")
print("=" * 70)

comparison = """
┌────────────────────┬──────────────────────────┬──────────────────────────┐
│                    │      LangGraph           │  LlamaIndex Workflows    │
├────────────────────┼──────────────────────────┼──────────────────────────┤
│ Mental model       │ Graph (nodes + edges)    │ Event-driven (pub/sub)   │
│ Define logic       │ graph.add_node()         │ @step decorator          │
│ Control flow       │ add_conditional_edges()  │ Return different Events  │
│ Shared state       │ TypedDict                │ Context (key-value)      │
│ State mutation     │ Reducer functions        │ ctx.set() / ctx.get()    │
│ Loops              │ Edge back to earlier node│ Emit consumed Event type │
│ Visualization      │ graph.get_graph().draw() │ draw_all_possible_flows()│
│ Async support      │ Full                     │ Native (async-first)     │
│ Checkpointing      │ Built-in (MemorySaver)   │ Via llama-deploy         │
│ Human-in-the-loop  │ interrupt_before/after   │ InputRequiredEvent       │
│ Best combined with │ LlamaIndex for retrieval │ LangGraph for complex    │
│                    │                          │ multi-agent orchestration│
└────────────────────┴──────────────────────────┴──────────────────────────┘

INTERVIEW ANSWER:
"LangGraph and LlamaIndex Workflows are two approaches to agent orchestration.
LangGraph uses a graph metaphor — nodes are functions, edges are transitions,
and you can visualize the entire flow. LlamaIndex Workflows use an event-driven
model — steps emit typed events that trigger other steps, like a pub/sub system.

I'd pick LangGraph when I need complex multi-agent coordination with cycles,
human-in-the-loop, and visual debugging via LangSmith. I'd pick LlamaIndex
Workflows when retrieval is the core logic and I want tight integration with
LlamaIndex's data abstractions (QueryEngines, Indexes, NodeParsers).

In practice, many production teams use LlamaIndex for the retrieval layer
inside a LangGraph orchestration layer — getting the best of both worlds."
"""

print(comparison)

print("=" * 70)
print("SUMMARY — LlamaIndex Workflows")
print("=" * 70)
print("""
What we built:
1. HELLO WORKFLOW — Minimal: StartEvent → @step → StopEvent
2. MULTI-STEP    — Custom Events chain steps together (classify → respond)
3. BRANCHING     — One step emits different events (simple vs complex path)
4. CONTEXT       — Shared state persists across steps (like LangGraph state)

Key patterns:
- @step(ev: EventType) → The event type hint = "run me when this event fires"
- Return StopEvent to finish the workflow
- Return a custom Event to trigger the next step
- Return EventA | EventB for branching (union type hint)
- Use Context for cross-step state

When to use Workflows vs LangGraph:
- Workflows: RAG-centric apps, already using LlamaIndex, prefer async Python
- LangGraph: Complex multi-agent systems, need visual debugging, prefer graphs
""")
