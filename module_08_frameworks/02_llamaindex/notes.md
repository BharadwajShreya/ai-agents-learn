# Module 8.2: LlamaIndex — Deep Dive Notes

## Table of Contents
1. [Why LlamaIndex?](#1-why-llamaindex)
2. [Core Abstractions](#2-core-abstractions)
3. [The RAG Pipeline — End to End](#3-the-rag-pipeline--end-to-end)
4. [Advanced Retrieval Strategies](#4-advanced-retrieval-strategies)
5. [Agentic RAG](#5-agentic-rag)
6. [Workflows API — Event-Driven Orchestration](#6-workflows-api--event-driven-orchestration)
7. [Production: llama-deploy](#7-production-llama-deploy)
8. [LlamaIndex vs LangChain/LangGraph](#8-llamaindex-vs-langchainlanggraph)
9. [Interview Questions & Answers](#9-interview-questions--answers)

---

## 1. Why LlamaIndex?

### THE PROBLEM
In Module 8.1, we learned LangGraph — great at orchestrating complex agent flows (state machines, conditional edges, cycles). But LangGraph doesn't help you with the **data** side:
- How do you chunk 10,000 PDFs intelligently?
- How do you embed and index them efficiently?
- How do you retrieve the RIGHT chunks for a query?
- How do you synthesize a grounded answer from retrieved chunks?

These are **retrieval engineering** problems, not orchestration problems.

### THE SOLUTION
LlamaIndex is a **data framework** purpose-built for retrieval-augmented generation. It provides:
- **10+ chunking strategies** (sentence, token, semantic, hierarchical)
- **Advanced retrieval** (auto-merging, sentence window, hybrid search)
- **Response synthesis modes** (compact, refine, tree summarize)
- **Agentic RAG** (agents that decide which knowledge base to query)
- **Production runtime** (llama-deploy for distributed deployments)

### THE MENTAL MODEL
```
LangGraph = orchestration framework (HOW to process)
LlamaIndex = data framework (WHAT to process)
```

LangGraph is about **control flow** — "first do A, then if X do B else do C."
LlamaIndex is about **data flow** — "load these docs, chunk them, embed them, retrieve the right ones, synthesize an answer."

### INTERVIEW ANGLE
> "I'd use LlamaIndex when the core challenge is retrieval quality — when the app's value depends on finding and synthesizing the right information from large document corpora. For complex multi-step agent orchestration with cycles, human-in-the-loop, and conditional branching, I'd use LangGraph. Many production systems use both: LlamaIndex as the retrieval layer inside a LangGraph orchestration layer."

**Code reference:** `01_core_abstractions.py` (pure Python demo)

---

## 2. Core Abstractions

### THE HIERARCHY
```
Raw Files (PDF, TXT, HTML, DB rows)
    │
    ▼
Documents — raw text + metadata wrappers
    │
    ▼ (NodeParser)
Nodes — chunks with metadata + relationships
    │
    ▼ (EmbeddingModel)
Index — nodes + embeddings stored for retrieval
    │
    ▼
Retriever — finds relevant nodes for a query
    │
    ▼
ResponseSynthesizer — combines nodes + query → LLM → answer
    │
    ▼
QueryEngine — wraps Retriever + Synthesizer in one call
```

### DOCUMENT
- Wrapper around text + metadata
- One per source file (one PDF page = one Document)
- Created by **Readers** (SimpleDirectoryReader, PDFReader, DatabaseReader, etc.)
- **Immutable** — you never modify a Document, you parse it into Nodes

```python
from llama_index.core.schema import Document

doc = Document(
    text="Revenue grew 23% year-over-year...",
    metadata={"source": "q1_report.pdf", "page": 3, "year": 2026}
)
```

### NODE
- **Atomic unit of retrieval** — what actually gets embedded and retrieved
- A chunk of a Document with its own metadata
- Has **relationships** — PREVIOUS, NEXT, SOURCE (parent document)
- Relationships enable context-aware retrieval (grab neighboring chunks)

```python
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

node = TextNode(
    text="Revenue grew 23%...",
    metadata={"source": "q1_report.pdf"},
)
# Link to parent document
node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc_001")
```

### NODE PARSERS (Chunking Strategies)
Different parsers = different tradeoffs:

| Parser | How it splits | Best for | Tradeoff |
|--------|--------------|----------|----------|
| `SentenceSplitter` | Sentence boundaries | General text, articles | Fast, respects grammar |
| `TokenTextSplitter` | Exact token count | Strict budget control | May break mid-sentence |
| `SemanticSplitter` | Embedding similarity | Topic-aware chunking | Slow (needs embeddings) |
| `SentenceWindowParser` | Per-sentence, with window | Fine-grained retrieval | More nodes to store |
| `HierarchicalNodeParser` | Multi-level (doc/section/chunk) | Long documents | Complex index structure |

**Key parameters:**
- `chunk_size`: Max tokens per chunk. Smaller = more precise, larger = more context.
- `chunk_overlap`: Tokens shared between adjacent chunks. Prevents context loss at boundaries. **10-20% of chunk_size is standard.**

### INDEX
Stores nodes + their embeddings for retrieval:

| Index Type | How it works | Best for |
|-----------|-------------|----------|
| `VectorStoreIndex` | Embeds all nodes, similarity search | Most RAG use cases (DEFAULT) |
| `SummaryIndex` | Stores all nodes, no embedding | When you need ALL context |
| `KeywordTableIndex` | Keyword extraction + lookup | Keyword-heavy queries |
| `KnowledgeGraphIndex` | Entities + relationships | Structured knowledge |

In production, `VectorStoreIndex` backed by a real vector DB (Chroma, Pinecone, Weaviate) is the standard.

### RETRIEVER
Finds the most relevant nodes for a query:
- `VectorIndexRetriever` — cosine similarity search (default)
- `BM25Retriever` — keyword-based (sparse retrieval)
- `HybridRetriever` — combines dense + sparse (best of both)
- `AutoMergingRetriever` — retrieves small chunks, merges into larger parents
- `SentenceWindowRetriever` — retrieves one sentence, returns surrounding window

**`similarity_top_k`** — how many nodes to retrieve. Typical values: 3-10.

### RESPONSE SYNTHESIZER
Combines retrieved nodes + query → LLM → answer:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `COMPACT` | Stuff as many nodes as possible into one prompt | Default. Fast, cheap. |
| `REFINE` | Build answer iteratively, one node at a time | Better quality, more LLM calls |
| `TREE_SUMMARIZE` | Build a tree of summaries | Large result sets |
| `SIMPLE_SUMMARIZE` | Truncate to fit, summarize | Fastest but may lose info |

### QUERY ENGINE
High-level wrapper: `QueryEngine = Retriever + ResponseSynthesizer`

```python
# One-liner
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is the PTO policy?")

# Or decompose for fine-grained control
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("What is the PTO policy?")
synthesizer = get_response_synthesizer(response_mode=ResponseMode.REFINE)
response = synthesizer.synthesize("What is the PTO policy?", nodes)
```

**Code reference:** `01_core_abstractions.py`, `02_rag_pipeline.py`

---

## 3. The RAG Pipeline — End to End

### INGESTION (Offline — happens once)
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Step 1: Load
documents = SimpleDirectoryReader("./data").load_data()

# Step 2: Index (chunks + embeds + stores — all in one!)
index = VectorStoreIndex.from_documents(documents)
```

That's it. Two lines. LlamaIndex handles:
1. Splitting documents into nodes (default: SentenceSplitter, chunk_size=1024)
2. Embedding each node (using the configured embedding model)
3. Storing in an in-memory vector store

### RETRIEVAL (Online — per query)
```python
# Option A: High-level (one call)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("your question here")

# Option B: Fine-grained control
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("your question")
# ... post-process, rerank, filter ...
synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
response = synthesizer.synthesize("your question", nodes)
```

### CUSTOMIZATION KNOBS
```python
from llama_index.core.node_parser import SentenceSplitter

# Custom chunking
custom_parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)

# Custom index
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[custom_parser],  # Override default chunking
)

# Custom query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,                    # How many chunks to retrieve
    response_mode=ResponseMode.REFINE,     # How to synthesize
)
```

### THE #1 TUNING KNOB
**Chunk size** is the single most impactful parameter:
- Too large (2000+ tokens): Retrieval gets imprecise (too much noise per chunk)
- Too small (50-100 tokens): Loses context (chunk doesn't have enough info)
- Sweet spot: **256-512 tokens** for most use cases

**Code reference:** `02_rag_pipeline.py`

---

## 4. Advanced Retrieval Strategies

### SENTENCE WINDOW RETRIEVAL
**Problem:** Small chunks = precise retrieval but insufficient context.
**Solution:** Store individual sentences, but when retrieved, return a WINDOW of surrounding sentences.

```
Index:  [sent1] [sent2] [sent3] [sent4] [sent5]
Query matches: sent3
Return: sent2 + sent3 + sent4 (window_size=1)
```

The retrieval is precise (matched one sentence), but the LLM gets enough context to answer well.

### AUTO-MERGING RETRIEVAL
**Problem:** Sometimes you need a bigger chunk, sometimes smaller.
**Solution:** Build a hierarchical index (document → section → paragraph → sentence). Retrieve at the leaf level, then merge up if enough leaves from the same parent are retrieved.

```
Document
  ├── Section 1
  │     ├── Para 1 ← retrieved
  │     ├── Para 2 ← retrieved
  │     └── Para 3
  └── Section 2
        ├── Para 4
        └── Para 5 ← retrieved

Since 2/3 paragraphs from Section 1 were retrieved,
auto-merge returns the entire Section 1 instead.
```

### HYBRID RETRIEVAL (BM25 + Dense)
**Problem:** Dense (embedding) retrieval is great for semantic meaning but misses exact keywords. Sparse (BM25) is great for exact matches but misses semantics.
**Solution:** Combine both and rerank.

```
Dense retrieval:  [doc3, doc7, doc1]  (semantic similarity)
Sparse retrieval: [doc1, doc5, doc3]  (keyword match)
Reranker:         [doc3, doc1, doc7, doc5]  (combined + reranked)
```

### RERANKING
After initial retrieval, use a cross-encoder model to rescore results.
Initial retrieval is fast but approximate (bi-encoder). Reranking is slow but accurate (cross-encoder).

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(top_n=3, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
query_engine = index.as_query_engine(
    similarity_top_k=10,           # Retrieve 10 initially
    node_postprocessors=[reranker],  # Rerank down to 3
)
```

### INTERVIEW ANGLE
> "For advanced retrieval, I'd start with Sentence Window retrieval if I need precision on specific facts, or Auto-Merging if documents have clear hierarchical structure. Hybrid retrieval (BM25 + dense) with reranking is the gold standard for production — it handles both semantic and keyword queries well. The reranker is the single biggest quality boost after getting chunking right."

---

## 5. Agentic RAG

### THE UPGRADE
```
Basic RAG:    query → one index → answer (static pipeline)
Agentic RAG:  query → agent decides → picks tool(s) → retrieves → answers
```

**Why this matters:** In real companies, knowledge lives in different systems — HR policies, engineering runbooks, product docs, financial reports. An agent can route questions to the right knowledge base.

### THE PATTERN
```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# Step 1: Build separate indexes
hr_index = VectorStoreIndex.from_documents(hr_docs)
eng_index = VectorStoreIndex.from_documents(eng_docs)

# Step 2: Wrap as tools (the DESCRIPTION is critical!)
hr_tool = QueryEngineTool(
    query_engine=hr_index.as_query_engine(),
    metadata=ToolMetadata(
        name="hr_policies",
        description="Use for questions about PTO, remote work, promotions, benefits."
    ),
)

eng_tool = QueryEngineTool(
    query_engine=eng_index.as_query_engine(),
    metadata=ToolMetadata(
        name="engineering_practices",
        description="Use for questions about code review, incidents, API design."
    ),
)

# Step 3: Create agent
agent = ReActAgent.from_tools([hr_tool, eng_tool], llm=llm, verbose=True)

# Step 4: Query — agent picks the right tool
response = agent.chat("What is the PTO policy?")  # → routes to hr_policies
```

### KEY DESIGN DECISION: TOOL DESCRIPTIONS
The tool **description** is the most important thing to get right. The agent uses it to decide which tool to call. Guidelines:
- **Be specific** about what the tool covers: "Covers PTO, remote work, sick leave..."
- **Be explicit** about when to use it: "Use for ANY question about employee benefits"
- **Include negative examples** if helpful: "Do NOT use for product feature questions"

### MULTI-TOOL QUERIES
For questions that span multiple knowledge bases ("Compare on-call pay with the equipment stipend"), the agent can call MULTIPLE tools in sequence:
1. Call `engineering_practices` → get on-call compensation info
2. Call `hr_policies` → get equipment stipend info
3. Synthesize both into a comparison

### AGENT TYPES IN LLAMAINDEX
| Agent Type | How it works | When to use |
|-----------|-------------|-------------|
| `ReActAgent` | Reason → Act → Observe loop | General purpose, versatile |
| `FunctionCallingAgentWorker` | Uses LLM's native tool-calling | Better with OpenAI/Gemini models |
| `Workflow-based Agent` | Custom event-driven logic | Complex, customized flows |

**Code reference:** `03_agentic_rag.py`

---

## 6. Workflows API — Event-Driven Orchestration

### WHY WORKFLOWS?
LlamaIndex Workflows are the **modern** replacement for older chain-based patterns. They use an event-driven architecture that's more flexible than rigid chains but more structured than raw agent loops.

### CORE CONCEPTS
| Concept | What it is | Analogy |
|---------|-----------|---------|
| `Workflow` | Container class for steps | The state machine itself |
| `@step` | Decorator on async methods | A node in LangGraph |
| `Event` | Typed data that triggers steps | An edge in LangGraph |
| `StartEvent` | Entry point | `START` node in LangGraph |
| `StopEvent` | Exit point (carries result) | `END` node in LangGraph |
| `Context` | Shared state across steps | The `State` TypedDict in LangGraph |

### BASIC PATTERN
```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event

class MyEvent(Event):
    data: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> MyEvent:
        return MyEvent(data="processed!")

    @step
    async def step_two(self, ev: MyEvent) -> StopEvent:
        return StopEvent(result=f"Done: {ev.data}")

# Run
w = MyWorkflow(timeout=30)
result = await w.run()
```

### BRANCHING (Conditional Routing)
Return different event types based on a condition:
```python
@step
async def router(self, ev: StartEvent) -> SimpleEvent | ComplexEvent:
    if is_simple(ev.get("query")):
        return SimpleEvent(query=ev.get("query"))
    else:
        return ComplexEvent(query=ev.get("query"))
```
This is the Workflow equivalent of `add_conditional_edges()` in LangGraph.

### LOOPING (Feedback Loops)
A step can emit an event that was consumed by a previous step:
```python
@step
async def generate(self, ev: StartEvent | RetryEvent) -> ValidateEvent:
    # This step handles BOTH the initial start AND retry events
    answer = await llm.acomplete(ev.get("query"))
    return ValidateEvent(answer=answer)

@step
async def validate(self, ev: ValidateEvent) -> StopEvent | RetryEvent:
    if is_valid(ev.answer):
        return StopEvent(result=ev.answer)
    else:
        return RetryEvent(query="try again...")  # Loops back to generate!
```
This is the Workflow equivalent of a cycle in LangGraph.

### CONTEXT (Shared State)
```python
@step
async def step_one(self, ctx: Context, ev: StartEvent) -> MyEvent:
    await ctx.set("counter", 1)
    return MyEvent(data="hello")

@step
async def step_two(self, ctx: Context, ev: MyEvent) -> StopEvent:
    counter = await ctx.get("counter")
    return StopEvent(result=f"Counter was {counter}")
```

### COMPARISON: LANGGRAPH vs WORKFLOWS

| Aspect | LangGraph | LlamaIndex Workflows |
|--------|-----------|---------------------|
| Mental model | "Drawing a flowchart" | "Writing event handlers" |
| Define logic | `graph.add_node(name, fn)` | `@step` decorator |
| Control flow | `add_conditional_edges()` | Return different Event types |
| State | TypedDict with reducers | Context (key-value store) |
| Loops | Edge back to earlier node | Emit consumed Event type |
| Visualization | `graph.get_graph().draw_mermaid()` | `draw_all_possible_flows()` |
| Checkpointing | Built-in MemorySaver | Via llama-deploy |
| Human-in-the-loop | `interrupt_before`/`interrupt_after` | `InputRequiredEvent` |
| Async | Supported | **Native** (async-first) |

**Code reference:** `04_workflows_demo.py`

---

## 7. Production: llama-deploy

### WHAT IS IT?
`llama-deploy` is LlamaIndex's production runtime for deploying Workflows as microservices. It takes the same Workflow code you write locally and runs it in a distributed, scalable environment.

### ARCHITECTURE
```
┌─────────────────────────────────────────────────────┐
│                    llama-deploy                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐                                    │
│  │ API Gateway   │ ← Client requests come in here    │
│  └──────┬───────┘                                    │
│         │                                            │
│  ┌──────▼───────┐                                    │
│  │ Control Plane │ ← The "brain" — routes tasks      │
│  │ (Orchestrator)│   manages sessions, handles errors │
│  └──────┬───────┘                                    │
│         │                                            │
│  ┌──────▼───────┐                                    │
│  │ Message Queue │ ← Communication backbone          │
│  │ (Redis/Kafka/ │   async task distribution          │
│  │  RabbitMQ/SQS)│                                    │
│  └──────┬───────┘                                    │
│         │                                            │
│  ┌──────▼───────────────────────────────────┐        │
│  │        Workflow Services                  │        │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │        │
│  │  │ RAG     │ │ Agent   │ │ Parser  │    │        │
│  │  │ Service │ │ Service │ │ Service │    │        │
│  │  └─────────┘ └─────────┘ └─────────┘    │        │
│  │  (Scale each independently)              │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### KEY COMPONENTS
- **Control Plane**: Routes tasks, manages sessions, handles retries
- **Message Queue**: Async communication (Redis, Kafka, RabbitMQ, AWS SQS)
- **Workflow Services**: Your actual Workflow code, running as independent services

### THE KEY BENEFIT
**Same code, local → production:**
```python
# LOCAL development (single process)
result = await my_workflow.run(query="...")

# PRODUCTION (distributed microservices via llama-deploy)
# Same Workflow class, deployed as a service
# Control plane handles routing, scaling, fault tolerance
```

### VS LANGSMITH/LANGGRAPH PLATFORM
| Feature | llama-deploy | LangGraph Platform |
|---------|-------------|-------------------|
| Focus | Deploying LlamaIndex Workflows | Deploying LangGraph agents |
| Observability | OpenInference (OTel) | LangSmith (proprietary) |
| Scaling | Independent microservices | Managed cloud |
| Message Queue | BYO (Redis, Kafka, etc.) | Built-in |
| Best for | RAG-heavy systems | Agent-heavy systems |

### INTERVIEW ANGLE
> "llama-deploy provides a production runtime for LlamaIndex that separates your workflow code from infrastructure concerns. The Control Plane handles routing and session management, the Message Queue handles async communication, and each Workflow Service can scale independently. This is important when you have different parts of your pipeline with different load characteristics — your retrieval service might need 10x more instances than your parsing service."

---

## 8. LlamaIndex vs LangChain/LangGraph

### THE 60-SECOND VERDICT
| Dimension | LlamaIndex | LangChain/LangGraph |
|-----------|-----------|-------------------|
| **Primary strength** | Data retrieval quality | Workflow orchestration |
| **Core metaphor** | Data pipeline | State machine |
| **Best for** | Knowledge-heavy apps (RAG, doc QA) | Tool-heavy agents (API calls, complex flows) |
| **Chunking** | 5+ built-in strategies | Basic (or BYO) |
| **Retrieval** | 10+ strategies, reranking, hybrid | BYO retrieval |
| **Response synthesis** | 4 modes (compact, refine, tree, simple) | Not its job |
| **Orchestration** | Workflows (event-driven) | StateGraph (graph-based) |
| **Observability** | OpenInference / Arize | LangSmith |
| **Production** | llama-deploy | LangGraph Platform |

### DECISION FRAMEWORK
```
START
  │
  ├─ Is retrieval quality the core challenge?
  │   YES → LlamaIndex
  │   │
  │   └─ Do you also need complex agent orchestration?
  │       YES → LlamaIndex for retrieval + LangGraph for orchestration
  │       NO  → LlamaIndex only (Workflows for simple orchestration)
  │
  └─ Is complex agent control flow the core challenge?
      YES → LangGraph
      │
      └─ Do you also need high-quality retrieval?
          YES → LangGraph for orchestration + LlamaIndex for retrieval
          NO  → LangGraph only
```

### USING BOTH (Common Production Pattern)
```python
# LlamaIndex handles retrieval
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

# LangGraph handles orchestration
from langgraph.graph import StateGraph

def retrieve_node(state):
    """LangGraph node that calls LlamaIndex under the hood."""
    result = query_engine.query(state["query"])
    return {"context": str(result)}

# Build LangGraph with LlamaIndex as a retrieval step
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("reason", reasoning_node)
graph.add_edge("retrieve", "reason")
```

### INTERVIEW ANGLE
> "The choice between LlamaIndex and LangGraph depends on where your app's complexity lies. If the hard part is finding and synthesizing the right information from large document corpora, LlamaIndex is the better choice — it has built-in chunking strategies, advanced retrieval (sentence window, auto-merging, hybrid), and response synthesis modes. If the hard part is orchestrating complex agent behavior with conditional branching, human-in-the-loop, and multi-agent coordination, LangGraph is better. Most serious production systems use both: LlamaIndex as the 'data brain' and LangGraph as the 'logic brain'."

---

## 9. Interview Questions & Answers

### Q1: What is LlamaIndex and how does it differ from LangChain?
**A:** LlamaIndex is a data framework for LLM applications, purpose-built for retrieval quality. While LangChain focuses on chaining LLM calls and tool use (with LangGraph for complex orchestration), LlamaIndex excels at the data side: ingesting documents, intelligent chunking, advanced retrieval strategies, and response synthesis. The core abstractions are different — LlamaIndex has Documents, Nodes, Indexes, Retrievers, and QueryEngines that form a data pipeline. LangChain has Chains, Agents, and Tools that form an execution pipeline. Many production systems use both: LlamaIndex for retrieval and LangChain/LangGraph for orchestration.

---

### Q2: Explain the end-to-end flow of a LlamaIndex QueryEngine.
**A:** A QueryEngine orchestrates: (1) The Retriever takes the user query, embeds it, and performs similarity search against the index to find the top-k most relevant Nodes. (2) Optional NodePostprocessors can rerank, filter, or deduplicate the retrieved nodes. (3) The ResponseSynthesizer takes the retrieved nodes and the original query, formats them into a prompt, and sends it to the LLM for a grounded answer. The result includes the response text and source nodes for citation. Under the hood, `index.as_query_engine()` wires all three components together.

---

### Q3: What chunking strategy would you use for a large technical document?
**A:** It depends on the document structure and retrieval needs:
- **SentenceSplitter** (default): Good starting point for most text. Respects sentence boundaries, configurable chunk_size (256-512 tokens is typical) with overlap.
- **HierarchicalNodeParser**: Best for well-structured docs (with headers, sections). Creates a multi-level index that supports auto-merging retrieval.
- **SemanticSplitter**: Best quality — uses embeddings to split at topic boundaries. Slower but produces the most coherent chunks. 
- Key parameters: chunk_size (256-512 typical), chunk_overlap (10-20% of chunk_size). The chunk_overlap prevents losing context at boundaries.

---

### Q4: What is Agentic RAG and how does it differ from basic RAG?
**A:** Basic RAG is a static pipeline: query → one index → answer. Agentic RAG adds an intelligent routing layer: the LLM agent reads the query, decides which knowledge base(s) to consult, retrieves from them, and synthesizes a cross-domain answer. This is implemented using QueryEngineTools — each wrapping a different index — given to a ReActAgent. The agent uses the tool descriptions to decide which tool to call. For cross-domain questions, the agent can call multiple tools in sequence. The key design decision is writing precise tool descriptions — the agent's accuracy depends entirely on understanding when to use each tool.

---

### Q5: How do LlamaIndex Workflows compare to LangGraph?
**A:** Both are orchestration systems but with different paradigms:
- **LangGraph** uses a graph metaphor: nodes are functions, edges are transitions, state is a TypedDict. You "draw" the flow with `add_node()` and `add_edge()`. Conditional routing uses `add_conditional_edges()`.
- **LlamaIndex Workflows** use an event-driven model: steps are `@step`-decorated async methods, events are typed data objects that trigger steps. Conditional routing = returning different Event types. State is managed via a Context key-value store.
- LangGraph excels at visual debugging (graph visualization, LangSmith). Workflows are async-first and integrate naturally with LlamaIndex's data abstractions.
- I'd pick LangGraph for complex multi-agent systems and Workflows for RAG-centric applications already using LlamaIndex.

---

### Q6: What is the role of metadata in LlamaIndex and why is it important?
**A:** Metadata serves as a pre-filter before vector similarity search. Each Node inherits metadata from its parent Document (file_name, page, etc.) and can have custom metadata (department, date, author). During retrieval, you can apply metadata filters ("only search finance docs from Q1 2026") to narrow the search space BEFORE running the embedding similarity search. This dramatically improves precision in enterprise settings where you have thousands of documents across departments. Without metadata filtering, the retriever might return semantically similar but irrelevant chunks from the wrong department or time period.

---

### Q7: Explain the different response synthesis modes.
**A:** LlamaIndex offers four modes:
1. **COMPACT** (default): Stuffs as many retrieved nodes as possible into a single prompt. Fast, cheap (one LLM call). Best for most cases.
2. **REFINE**: Sends nodes to the LLM one at a time, asking it to refine its answer with each new node. Better quality for complex questions, but multiple LLM calls = slower and more expensive.
3. **TREE_SUMMARIZE**: Builds a tree of summaries from the nodes, then summarizes the summaries. Best for large result sets (20+ nodes).
4. **SIMPLE_SUMMARIZE**: Truncates nodes to fit the context window, then summarizes. Fastest but may lose information.

The choice depends on your quality-vs-cost tradeoff: COMPACT for speed, REFINE for quality, TREE_SUMMARIZE for scale.

---

### Q8: How would you debug a RAG system that returns hallucinated answers?
**A:** I'd investigate each pipeline stage:
1. **Retrieval**: Are the right chunks being retrieved? Check `retriever.retrieve(query)` directly. If chunks are irrelevant, tune chunk_size, try hybrid retrieval (BM25 + dense), or add a reranker.
2. **Context quality**: Even if the right chunks are retrieved, are they providing enough context? Try sentence window retrieval or increase top_k.
3. **Synthesis**: Is the LLM ignoring the context? Adjust the prompt template to force grounding ("Only answer based on the provided context. If the context doesn't contain the answer, say 'I don't know'").
4. **Evaluation**: Use metrics like Faithfulness (does the answer align with retrieved context?) and Relevancy (are retrieved nodes relevant to the query?) from frameworks like RAGAS.
5. **Metadata**: Add metadata filters to ensure the retriever searches the right document subset.

---

### Q9: What is llama-deploy and when would you use it?
**A:** llama-deploy is LlamaIndex's production runtime that turns Workflows into distributed microservices. It has three components: (1) Control Plane — routes tasks, manages sessions, handles errors; (2) Message Queue — async communication backbone (supports Redis, Kafka, RabbitMQ, SQS); (3) Workflow Services — your actual code running as independent, scalable services. The key benefit is "same code, local → production" — your Workflow class runs locally during development and deploys as a microservice in production without code changes. I'd use it when I need to scale different parts of my pipeline independently (e.g., retrieval service needs 10x more instances than parsing).

---

### Q10: When would you use LlamaIndex alone vs. with LangGraph?
**A:** LlamaIndex alone is sufficient when:
- The app is primarily about knowledge retrieval and question answering
- The control flow is simple (query → retrieve → answer, or route → retrieve → answer)
- You can express any needed orchestration with Workflows

Add LangGraph when you need:
- Complex multi-agent coordination (supervisor → workers)
- Human-in-the-loop approval at specific steps
- Persistent checkpointing and resume-after-crash
- Cycles with conditional exits that are hard to express as events
- Visual debugging and monitoring via LangSmith

The common production pattern is: LlamaIndex handles the "data brain" (retrieval, chunking, synthesis) while LangGraph handles the "logic brain" (orchestration, routing, multi-agent coordination).
