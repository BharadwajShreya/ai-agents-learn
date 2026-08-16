"""
Module 8.2 — Script 03: Agentic RAG with LlamaIndex
====================================================
The BIG UPGRADE from basic RAG:
  Basic RAG:   query → one index → answer
  Agentic RAG: query → agent DECIDES which index(es) to query → answer

This is like giving a research assistant access to multiple filing cabinets
instead of just one. The agent reads your question and picks the right cabinet.

WHY THIS MATTERS:
In real companies, knowledge lives in different systems:
  - HR policies (Confluence)
  - Product docs (Notion)
  - Engineering runbooks (GitHub Wiki)
  - Financial reports (SharePoint)

An agentic RAG system can route "What's our PTO policy?" to HR docs
and "How do we handle SEV-1 incidents?" to engineering runbooks.

Run: source ../venv/bin/activate && python 03_agentic_rag.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================================
# STEP 0: Configure LLM and Embeddings
# ============================================================================
llm = OpenAI(
    model="google/gemma-3-27b-it:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = llm
Settings.embed_model = embed_model

print("✅ LLM + Embeddings configured")
print()


# ============================================================================
# STEP 1: Build SEPARATE indexes for different knowledge domains
# ============================================================================
# This is the key difference from basic RAG — multiple indexes, not one.

print("=" * 70)
print("STEP 1: Building Separate Knowledge Bases")
print("=" * 70)

data_dir = Path(__file__).parent / "sample_data"

# --- Knowledge Base 1: Company Policies (HR) ---
policies_docs = SimpleDirectoryReader(
    input_files=[str(data_dir / "company_policies.txt")]
).load_data()
policies_index = VectorStoreIndex.from_documents(policies_docs)
print(f"📁 HR Policies index: {len(list(policies_index.docstore.docs.values()))} nodes")

# --- Knowledge Base 2: Product Documentation ---
product_docs = SimpleDirectoryReader(
    input_files=[str(data_dir / "product_docs.txt")]
).load_data()
product_index = VectorStoreIndex.from_documents(product_docs)
print(f"📁 Product Docs index: {len(list(product_index.docstore.docs.values()))} nodes")

# --- Knowledge Base 3: Engineering Practices ---
engineering_docs = SimpleDirectoryReader(
    input_files=[str(data_dir / "engineering_practices.txt")]
).load_data()
engineering_index = VectorStoreIndex.from_documents(engineering_docs)
print(f"📁 Engineering Practices index: {len(list(engineering_index.docstore.docs.values()))} nodes")

print()


# ============================================================================
# STEP 2: Wrap each index as a TOOL for the agent
# ============================================================================
# QueryEngineTool turns a QueryEngine into a callable tool.
# The DESCRIPTION is critical — the agent uses it to decide WHICH tool to call.
# A bad description = the agent picks the wrong knowledge base.

print("=" * 70)
print("STEP 2: Creating QueryEngine Tools")
print("=" * 70)

policies_tool = QueryEngineTool(
    query_engine=policies_index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="hr_policies",
        description=(
            "Provides information about company HR policies including: "
            "remote work policy, PTO/vacation policy, sick leave, "
            "performance reviews, promotion criteria, and equipment stipends. "
            "Use this tool for ANY question about company rules, employee benefits, "
            "or workplace policies."
        ),
    ),
)

product_tool = QueryEngineTool(
    query_engine=product_index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="product_docs",
        description=(
            "Provides information about TechNova's product NovaMail — "
            "an AI email assistant. Covers features (Smart Compose, summarization, "
            "Priority Inbox), technical architecture, API rate limits, pricing tiers, "
            "known limitations, and product roadmap. "
            "Use this tool for ANY question about the product or its capabilities."
        ),
    ),
)

engineering_tool = QueryEngineTool(
    query_engine=engineering_index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="engineering_practices",
        description=(
            "Provides information about engineering best practices including: "
            "code review guidelines, incident response playbook (SEV levels), "
            "on-call rotation and compensation, API design standards, "
            "rate limiting, and authentication patterns. "
            "Use this tool for ANY question about engineering processes or standards."
        ),
    ),
)

tools = [policies_tool, product_tool, engineering_tool]
print(f"✅ Created {len(tools)} QueryEngine tools:")
for tool in tools:
    print(f"   🔧 {tool.metadata.name}: {tool.metadata.description[:60]}...")
print()


# ============================================================================
# STEP 3: Create the Agent
# ============================================================================
# ReActAgent is LlamaIndex's built-in agent that uses the ReAct pattern.
# It reasons about WHICH tool to call, calls it, observes the result,
# and decides if it has enough info to answer.
#
# This is the EXACT same pattern we built manually in Module 7 and
# used with LangGraph in Module 8.1 — but LlamaIndex wires it up for RAG.

print("=" * 70)
print("STEP 3: Creating the ReAct Agent")
print("=" * 70)

agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,  # Show the agent's thought process (Thought → Action → Observation)
    max_iterations=5,  # Safety limit
)

print("✅ ReAct Agent created with 3 knowledge base tools")
print()


# ============================================================================
# STEP 4: Query the Agent — Watch it pick the right tool!
# ============================================================================
print("=" * 70)
print("STEP 4: Querying the Agent")
print("=" * 70)

# --- Query 1: Should route to HR policies ---
query1 = "How many days of PTO do employees get?"
print(f"\n{'─' * 60}")
print(f"📝 Query 1: \"{query1}\"")
print(f"   Expected tool: hr_policies")
print(f"{'─' * 60}")
response1 = agent.chat(query1)
print(f"\n🤖 Final Answer: {response1.response}")

# --- Query 2: Should route to product docs ---
query2 = "What model does NovaMail use and what is its latency?"
print(f"\n{'─' * 60}")
print(f"📝 Query 2: \"{query2}\"")
print(f"   Expected tool: product_docs")
print(f"{'─' * 60}")
response2 = agent.chat(query2)
print(f"\n🤖 Final Answer: {response2.response}")

# --- Query 3: Should route to engineering practices ---
query3 = "What happens during a SEV-1 incident?"
print(f"\n{'─' * 60}")
print(f"📝 Query 3: \"{query3}\"")
print(f"   Expected tool: engineering_practices")
print(f"{'─' * 60}")
response3 = agent.chat(query3)
print(f"\n🤖 Final Answer: {response3.response}")

# --- Query 4: Might need MULTIPLE tools ---
query4 = "Compare the on-call compensation with the remote work equipment stipend."
print(f"\n{'─' * 60}")
print(f"📝 Query 4 (multi-tool): \"{query4}\"")
print(f"   Expected tools: engineering_practices + hr_policies")
print(f"{'─' * 60}")
response4 = agent.chat(query4)
print(f"\n🤖 Final Answer: {response4.response}")


# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 70)
print("SUMMARY — Agentic RAG")
print("=" * 70)
print("""
What we built:
  3 SEPARATE knowledge bases (HR, Product, Engineering)
  3 QueryEngineTools wrapping each knowledge base
  1 ReAct Agent that DECIDES which tool(s) to call

Basic RAG vs Agentic RAG:
┌──────────────────────────────────────────────────────────────┐
│  BASIC RAG           │  AGENTIC RAG                          │
│  query → one index   │  query → agent picks index(es)        │
│  → answer            │  → retrieves → maybe queries another  │
│                      │  → synthesizes final answer            │
│  Static pipeline     │  Dynamic, intelligent routing          │
│  One knowledge base  │  Multiple knowledge bases              │
│  No reasoning        │  ReAct reasoning loop                  │
└──────────────────────────────────────────────────────────────┘

INTERVIEW ANSWER:
"Agentic RAG upgrades a static retrieval pipeline into an intelligent 
routing system. The agent reads the query, decides which knowledge base 
to consult, and can even query MULTIPLE sources for cross-domain questions. 
This is essential in enterprise settings where knowledge lives in 
different systems — HR, engineering, product, finance."

KEY DESIGN DECISION:
The tool DESCRIPTION is the most important thing to get right.
The agent uses the description to decide which tool to call.
A vague description → the agent picks the wrong tool → wrong answer.

Next: 04_workflows_demo.py — LlamaIndex's modern orchestration layer!
""")
