"""
Module 8.2 — Script 01: LlamaIndex Core Abstractions
=====================================================
NO LLM NEEDED — This script demonstrates the mental model of LlamaIndex
using pure Python. Understand these building blocks before touching real APIs.

THE KEY INSIGHT:
LlamaIndex is built around a data pipeline:
  Documents → Nodes → Index → Retriever → QueryEngine → Response

Each piece has ONE job. This is fundamentally different from LangGraph,
which is about orchestrating CONTROL FLOW (state machines, conditional edges).
LlamaIndex is about orchestrating DATA FLOW (ingestion → retrieval → synthesis).

Run: python 01_core_abstractions.py
Dependencies: pip install llama-index-core
"""

from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter


# ============================================================================
# PART 1: DOCUMENTS — The raw input
# ============================================================================
# A Document is just a wrapper around text + metadata.
# Think of it as: "one source file" — a PDF page, a web page, a database row.

print("=" * 70)
print("PART 1: DOCUMENTS")
print("=" * 70)

# Create documents manually (in real apps, you'd use SimpleDirectoryReader)
doc1 = Document(
    text="LlamaIndex is a data framework for LLM applications. "
         "It provides tools to ingest, structure, and access private data. "
         "The framework excels at retrieval-augmented generation (RAG).",
    metadata={
        "source": "official_docs",
        "category": "overview",
        "version": "2026"
    },
    doc_id="doc_001"
)

doc2 = Document(
    text="LangGraph models agents as state machines using directed graphs. "
         "Nodes are functions, edges are transitions, and state flows through "
         "the graph. It excels at complex orchestration with cycles and branches.",
    metadata={
        "source": "comparison_guide",
        "category": "frameworks",
        "version": "2026"
    },
    doc_id="doc_002"
)

print(f"Document 1 ID: {doc1.doc_id}")
print(f"Document 1 text (first 80 chars): {doc1.text[:80]}...")
print(f"Document 1 metadata: {doc1.metadata}")
print(f"Document 1 type: {type(doc1)}")
print()

# KEY INTERVIEW POINT:
# Documents are IMMUTABLE containers. They hold raw text + metadata.
# You never modify a Document — you parse it INTO Nodes.


# ============================================================================
# PART 2: NODES — The atomic units of retrieval
# ============================================================================
# A Node is a CHUNK of a Document. It's what actually gets embedded and retrieved.
# The Document → Node relationship is like a book → paragraphs.
# 
# WHY NODES AND NOT DOCUMENTS?
# - LLMs have limited context windows
# - Retrieval is more precise at the chunk level
# - You can track which chunk came from which document (provenance)

print("=" * 70)
print("PART 2: NODES (Manual Creation)")
print("=" * 70)

# Create nodes manually to understand the structure
node1 = TextNode(
    text="LlamaIndex is a data framework for LLM applications.",
    id_="node_001",
    metadata={"source": "official_docs", "chunk_index": 0}
)

node2 = TextNode(
    text="It provides tools to ingest, structure, and access private data.",
    id_="node_002",
    metadata={"source": "official_docs", "chunk_index": 1}
)

node3 = TextNode(
    text="The framework excels at retrieval-augmented generation (RAG).",
    id_="node_003",
    metadata={"source": "official_docs", "chunk_index": 2}
)

# Nodes can have RELATIONSHIPS — this is powerful for context
# "node2 came AFTER node1" — so if we retrieve node2, we can also grab node1 for context
node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id="node_001")
node2.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id="node_003")
node2.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc_001")

print(f"Node 2 text: {node2.text}")
print(f"Node 2 ID: {node2.id_}")
print(f"Node 2 relationships: {node2.relationships}")
print()

# KEY INTERVIEW POINT:
# Node relationships enable CONTEXT-AWARE retrieval.
# If you retrieve a node, you can "walk" to its neighbors for more context.
# This is the foundation of "Sentence Window" and "Auto-Merging" retrieval.


# ============================================================================
# PART 3: NODE PARSERS — Automatic chunking strategies
# ============================================================================
# In real apps, you don't create nodes manually. NodeParsers do it for you.
# Different parsers = different chunking strategies.

print("=" * 70)
print("PART 3: NODE PARSERS (Automatic Chunking)")
print("=" * 70)

sample_text = (
    "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs "
    "by grounding them in external knowledge. The process has three stages. "
    "First, relevant documents are retrieved from a knowledge base using semantic search. "
    "Second, the retrieved documents are injected into the LLM's prompt as context. "
    "Third, the LLM generates a response grounded in the provided context. "
    "RAG reduces hallucinations because the model can cite specific sources. "
    "It also keeps the model's knowledge current without expensive retraining. "
    "However, RAG quality depends heavily on retrieval quality — garbage in, garbage out. "
    "Advanced techniques like reranking, hybrid search, and query decomposition "
    "can significantly improve retrieval precision and recall."
)

sample_doc = Document(text=sample_text, metadata={"source": "rag_tutorial"})

# --- Strategy 1: SentenceSplitter ---
# Splits on sentence boundaries. Respects natural language structure.
# chunk_size = max tokens per chunk, chunk_overlap = tokens shared between chunks
sentence_parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)
sentence_nodes = sentence_parser.get_nodes_from_documents([sample_doc])

print("Strategy 1: SentenceSplitter (chunk_size=100, overlap=20)")
print(f"  Number of chunks: {len(sentence_nodes)}")
for i, node in enumerate(sentence_nodes):
    print(f"  Chunk {i}: ({len(node.text)} chars) \"{node.text[:70]}...\"")
print()

# --- Strategy 2: TokenTextSplitter ---
# Splits on exact token boundaries. More predictable size but may break mid-sentence.
token_parser = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
token_nodes = token_parser.get_nodes_from_documents([sample_doc])

print("Strategy 2: TokenTextSplitter (chunk_size=50, overlap=10)")
print(f"  Number of chunks: {len(token_nodes)}")
for i, node in enumerate(token_nodes):
    print(f"  Chunk {i}: ({len(node.text)} chars) \"{node.text[:70]}...\"")
print()

# KEY INTERVIEW POINT:
# "What chunking strategy would you use?"
# → SentenceSplitter for natural language docs (preserves sentence boundaries)
# → TokenTextSplitter when you need strict token budget control
# → SemanticSplitter (not shown — uses embeddings to split at topic boundaries)
#   → Best quality but slowest, requires an embedding model
#
# chunk_overlap is CRITICAL — without it, context at chunk boundaries is lost.
# A 10-20% overlap is standard practice.


# ============================================================================
# PART 4: METADATA — The secret weapon for production RAG
# ============================================================================
# Metadata filters let you narrow retrieval BEFORE semantic search.
# "Find relevant chunks, BUT ONLY from the Q1 2026 financial report."

print("=" * 70)
print("PART 4: METADATA FILTERING")
print("=" * 70)

docs_with_metadata = [
    Document(
        text="Revenue grew 23% year-over-year to $4.2B in Q1 2026.",
        metadata={"department": "finance", "quarter": "Q1", "year": 2026}
    ),
    Document(
        text="We hired 340 engineers in Q1, bringing total headcount to 12,500.",
        metadata={"department": "hr", "quarter": "Q1", "year": 2026}
    ),
    Document(
        text="Customer churn dropped to 2.1%, the lowest in company history.",
        metadata={"department": "customer_success", "quarter": "Q1", "year": 2026}
    ),
    Document(
        text="The new ML pipeline reduced inference costs by 40%.",
        metadata={"department": "engineering", "quarter": "Q1", "year": 2026}
    ),
]

parser = SentenceSplitter(chunk_size=256)
nodes = parser.get_nodes_from_documents(docs_with_metadata)

print("Nodes with metadata:")
for node in nodes:
    print(f"  [{node.metadata['department']}] {node.text[:60]}...")

# In a real system, you'd filter like:
# retriever.retrieve(query, filters={"department": "finance"})
# This DRAMATICALLY improves precision in enterprise RAG systems.

print()
print("KEY INSIGHT: Metadata filtering is a PRE-FILTER before vector search.")
print("It narrows the search space, improving both speed and relevance.")
print()


# ============================================================================
# PART 5: THE MENTAL MODEL — How everything connects
# ============================================================================
print("=" * 70)
print("PART 5: THE COMPLETE PIPELINE (Mental Model)")
print("=" * 70)

pipeline_diagram = """
┌─────────────────────────────────────────────────────────────────┐
│                    LlamaIndex RAG Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INGESTION (Offline — happens once)                              │
│  ─────────────────────────────────────                           │
│  Raw Files (PDF, TXT, HTML, DB)                                  │
│       │                                                          │
│       ▼                                                          │
│  Documents  ←── SimpleDirectoryReader / custom loaders           │
│       │                                                          │
│       ▼                                                          │
│  NodeParser  ←── SentenceSplitter / SemanticSplitter             │
│       │                                                          │
│       ▼                                                          │
│  Nodes (chunks + metadata + relationships)                       │
│       │                                                          │
│       ▼                                                          │
│  EmbeddingModel  ←── text-embedding-3-small / BGE / Gemini       │
│       │                                                          │
│       ▼                                                          │
│  VectorStoreIndex  ←── stored in Chroma / Pinecone / in-memory   │
│                                                                  │
│                                                                  │
│  RETRIEVAL (Online — happens per query)                          │
│  ─────────────────────────────────────                           │
│  User Query                                                      │
│       │                                                          │
│       ▼                                                          │
│  Retriever  ←── top-k similarity search (+ metadata filters)    │
│       │                                                          │
│       ▼                                                          │
│  NodePostprocessor  ←── reranking, filtering, dedup              │
│       │                                                          │
│       ▼                                                          │
│  ResponseSynthesizer  ←── compact / refine / tree_summarize      │
│       │                                                          │
│       ▼                                                          │
│  Response  ←── grounded answer with source citations             │
│                                                                  │
│                                                                  │
│  HIGH-LEVEL WRAPPER                                              │
│  ─────────────────                                               │
│  QueryEngine = Retriever + PostProcessor + ResponseSynthesizer   │
│  (One function call: query_engine.query("your question"))        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

print(pipeline_diagram)


# ============================================================================
# PART 6: LlamaIndex vs LangGraph — Different tools, different jobs
# ============================================================================
print("=" * 70)
print("PART 6: LlamaIndex vs LangGraph (Comparison)")
print("=" * 70)

comparison = """
┌────────────────────┬──────────────────────────┬──────────────────────────┐
│                    │      LlamaIndex          │      LangGraph           │
├────────────────────┼──────────────────────────┼──────────────────────────┤
│ Core metaphor      │ Data pipeline            │ State machine (graph)    │
│ Primary strength   │ Retrieval quality        │ Orchestration / control  │
│ Key primitive      │ QueryEngine              │ StateGraph               │
│ Best for           │ Knowledge-heavy apps     │ Tool-heavy agents        │
│ Chunking           │ Built-in, advanced       │ Not its job              │
│ Embeddings         │ First-class support      │ Not its job              │
│ Retrieval          │ 10+ strategies built-in  │ BYO retrieval            │
│ Agent patterns     │ Agentic RAG, Workflows   │ ReAct, Plan-Execute      │
│ State management   │ Context object           │ TypedDict state          │
│ When to combine    │ Use as retrieval layer   │ Use as orchestration     │
│ Production tooling │ llama-deploy             │ LangSmith                │
└────────────────────┴──────────────────────────┴──────────────────────────┘

INTERVIEW ANSWER:
"LlamaIndex and LangGraph solve different problems. LlamaIndex excels when 
your app's value comes from retrieval quality — it has built-in chunking, 
10+ retrieval strategies, and response synthesis modes. LangGraph excels 
when your app needs complex control flow — conditional branching, cycles, 
human-in-the-loop, and persistent checkpointing. In production, many teams 
use both: LlamaIndex as the retrieval layer inside a LangGraph state machine."
"""

print(comparison)


# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY — What we learned (no LLM needed!)")
print("=" * 70)
print("""
1. DOCUMENT   — Raw text + metadata wrapper. One per source file.
2. NODE       — Chunk of a Document. What actually gets embedded & retrieved.
3. NODEPARSER — Splits Documents into Nodes. Key strategies:
                SentenceSplitter, TokenTextSplitter, SemanticSplitter
4. INDEX      — Stores Nodes + embeddings. VectorStoreIndex is the default.
5. RETRIEVER  — Finds relevant Nodes for a query. Top-k similarity search.
6. RESPONSE SYNTHESIZER — Combines retrieved Nodes + query → LLM → answer.
7. QUERY ENGINE — High-level wrapper: Retriever + Synthesizer in one call.
8. METADATA   — Pre-filters search space before vector similarity.

Next: 02_rag_pipeline.py — build a full pipeline with a real LLM!
""")
