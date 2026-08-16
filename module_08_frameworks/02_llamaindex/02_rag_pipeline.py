"""
Module 8.2 — Script 02: Full RAG Pipeline with LlamaIndex
==========================================================
This script builds a complete RAG pipeline from scratch:
  Load documents → Parse into nodes → Build index → Query

We use:
  - OpenRouter (OpenAI-compatible) for the LLM
  - HuggingFace embeddings (free, local — no API key needed)
  - In-memory vector store (no external database)

This is the CORE use case of LlamaIndex — and what makes it different
from LangGraph. LangGraph can't do any of this natively.

Run: source ../venv/bin/activate && python 02_rag_pipeline.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from parent directory's .env
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# STEP 0: Configure LLM and Embeddings
# ============================================================================
# LlamaIndex uses a "Settings" singleton to configure defaults.
# Once set, every component (QueryEngine, Agent, etc.) uses these.

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LLM via OpenRouter (OpenAI-compatible API)
llm = OpenAI(
    model="google/gemma-3-27b-it:free",          # Free model on OpenRouter
    api_base="https://openrouter.ai/api/v1",      # OpenRouter endpoint
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,                               # Low temp for factual answers
)

# Embeddings via HuggingFace (runs locally — FREE, no API key)
# all-MiniLM-L6-v2 is small (~80MB), fast, and good enough for demos.
# In production, you'd use text-embedding-3-large or Gemini embeddings.
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set as global defaults — every LlamaIndex component will use these
Settings.llm = llm
Settings.embed_model = embed_model

print("✅ LLM: google/gemma-3-27b-it:free (via OpenRouter)")
print("✅ Embeddings: all-MiniLM-L6-v2 (local HuggingFace)")
print()


# ============================================================================
# STEP 1: Load Documents
# ============================================================================
# SimpleDirectoryReader reads all files in a directory.
# Supports: .txt, .pdf, .docx, .csv, .html, .md, and more.
# Each file becomes one Document object.

from llama_index.core import SimpleDirectoryReader

print("=" * 70)
print("STEP 1: Loading Documents")
print("=" * 70)

data_dir = Path(__file__).parent / "sample_data"
reader = SimpleDirectoryReader(input_dir=str(data_dir))
documents = reader.load_data()

print(f"Loaded {len(documents)} documents from {data_dir}")
for doc in documents:
    filename = doc.metadata.get("file_name", "unknown")
    print(f"  📄 {filename} — {len(doc.text)} chars")
print()

# KEY POINT: Each .txt file becomes ONE Document.
# The metadata automatically includes file_name, file_path, etc.


# ============================================================================
# STEP 2: Build the Index (Chunking + Embedding happens here)
# ============================================================================
# VectorStoreIndex does 3 things in one call:
#   1. Chunks documents into Nodes (using the default SentenceSplitter)
#   2. Embeds each Node using the configured embedding model
#   3. Stores everything in an in-memory vector store
#
# This is LlamaIndex's "killer feature" — what would take 50+ lines in
# raw Python (chunk, embed, store) is ONE line here.

from llama_index.core import VectorStoreIndex

print("=" * 70)
print("STEP 2: Building VectorStoreIndex")
print("=" * 70)

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
    # You can customize chunking here:
    # transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)]
)

# Let's inspect what happened under the hood
docstore = index.docstore
all_nodes = list(docstore.docs.values())
print(f"\n✅ Index built! {len(all_nodes)} nodes created from {len(documents)} documents")
print(f"   (Documents were automatically chunked into smaller nodes)")
print()

# Show a few nodes
for i, node in enumerate(all_nodes[:3]):
    source = node.metadata.get("file_name", "unknown")
    print(f"  Node {i}: [{source}] \"{node.text[:80]}...\"")
print(f"  ... and {len(all_nodes) - 3} more nodes")
print()


# ============================================================================
# STEP 3: Query the Index — Basic Retrieval + Synthesis
# ============================================================================
# QueryEngine = Retriever + ResponseSynthesizer
# One call: query → retrieve relevant nodes → synthesize answer

print("=" * 70)
print("STEP 3: Querying with QueryEngine")
print("=" * 70)

query_engine = index.as_query_engine(
    similarity_top_k=3,  # Retrieve top 3 most relevant chunks
)

# --- Query 1: Company policies ---
query1 = "What is the remote work policy?"
print(f"\n📝 Query: \"{query1}\"")
response1 = query_engine.query(query1)
print(f"\n🤖 Answer: {response1.response}")

# Show which source nodes were used
print(f"\n📎 Sources used ({len(response1.source_nodes)} nodes):")
for node in response1.source_nodes:
    score = node.score
    source = node.metadata.get("file_name", "unknown")
    print(f"  [{score:.4f}] {source}: \"{node.text[:60]}...\"")

print()

# --- Query 2: Product features ---
query2 = "What are the rate limits for the NovaMail API?"
print(f"📝 Query: \"{query2}\"")
response2 = query_engine.query(query2)
print(f"\n🤖 Answer: {response2.response}")
print()


# ============================================================================
# STEP 4: Under the Hood — Retriever and Synthesizer Separately
# ============================================================================
# Sometimes you want fine-grained control. Let's use them independently.

from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

print("=" * 70)
print("STEP 4: Retriever + Synthesizer (Separate)")
print("=" * 70)

# Get just the retriever
retriever = index.as_retriever(similarity_top_k=5)

# Retrieve relevant nodes (no LLM call yet!)
query3 = "How does the code review process work?"
retrieved_nodes = retriever.retrieve(query3)

print(f"\n📝 Query: \"{query3}\"")
print(f"\n🔍 Retrieved {len(retrieved_nodes)} nodes (no LLM call yet!):")
for i, node in enumerate(retrieved_nodes):
    print(f"  [{node.score:.4f}] {node.text[:80]}...")

# Now synthesize with different modes
# COMPACT mode: stuffs as many nodes as possible into one LLM call
synthesizer_compact = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
response_compact = synthesizer_compact.synthesize(query3, retrieved_nodes)
print(f"\n🤖 COMPACT mode answer: {response_compact.response[:200]}...")

# KEY INTERVIEW POINT:
# Response synthesis modes:
#   COMPACT  — Stuff everything into one prompt. Fast, cheap. Default.
#   REFINE   — Build answer iteratively, node by node. Better quality, more LLM calls.
#   TREE_SUMMARIZE — Build a tree of summaries. Best for large result sets.
#   SIMPLE_SUMMARIZE — Truncate to fit context, summarize. Fastest but may lose info.
print()


# ============================================================================
# STEP 5: Customizing the Pipeline
# ============================================================================
# Real production systems customize every step. Here's how.

from llama_index.core.node_parser import SentenceSplitter

print("=" * 70)
print("STEP 5: Custom Pipeline Configuration")
print("=" * 70)

# Custom chunking — smaller chunks for more precise retrieval
custom_parser = SentenceSplitter(
    chunk_size=256,       # Smaller chunks = more precise retrieval
    chunk_overlap=30,     # 30 token overlap between chunks
)

# Rebuild index with custom settings
custom_index = VectorStoreIndex.from_documents(
    documents,
    transformations=[custom_parser],
    show_progress=True,
)

custom_nodes = list(custom_index.docstore.docs.values())
print(f"\n✅ Custom index: {len(custom_nodes)} nodes (vs {len(all_nodes)} with defaults)")
print(f"   Smaller chunks = more nodes = more precise retrieval")

# Query with the custom index
custom_qe = custom_index.as_query_engine(similarity_top_k=3)
response_custom = custom_qe.query("What is the PTO policy?")
print(f"\n📝 Query: \"What is the PTO policy?\"")
print(f"🤖 Answer: {response_custom.response}")
print()


# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY — Full RAG Pipeline")
print("=" * 70)
print("""
What we built:
1. LOADED 3 documents using SimpleDirectoryReader
2. INDEXED them with VectorStoreIndex (auto-chunks + embeds + stores)
3. QUERIED using QueryEngine (retrieves + synthesizes in one call)
4. DECOMPOSED into Retriever + Synthesizer for fine-grained control
5. CUSTOMIZED the pipeline with different chunking parameters

Key takeaways:
- VectorStoreIndex.from_documents() is the 1-liner that does ingestion
- QueryEngine wraps Retriever + ResponseSynthesizer
- similarity_top_k controls how many chunks are retrieved
- ResponseMode controls how chunks are combined into an answer
- Chunk size is the #1 knob to tune for retrieval quality

Next: 03_agentic_rag.py — upgrade from static RAG to AGENTIC RAG!
""")
