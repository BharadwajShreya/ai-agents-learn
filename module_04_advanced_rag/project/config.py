"""
Enterprise RAG System — Centralized Configuration
===================================================

DESIGN DECISION: Why a centralized config?

In a real system, configuration is scattered across files:
  - Model names in one file
  - Chunk sizes in another
  - API keys hardcoded in a third

This causes two problems:
  1. DEBUGGING: "Which chunk size am I using?" → grep through 10 files
  2. EXPERIMENTATION: "Let me try chunk_size=1000" → change in 5 places

A centralized config solves both: ONE file to check, ONE place to change.

For production, this would be:
  - YAML/TOML config file (not Python — so non-devs can edit it)
  - Environment-based overrides (dev/staging/prod)
  - Feature flags for A/B testing different strategies
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure Windows stdout handles UTF-8 characters and emojis properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Load environment variables from .env file
# DESIGN DECISION: We look for .env in the project directory first,
# then fall back to the module_08 .env (where the key was originally set up).
_project_dir = Path(__file__).parent
_module08_env = _project_dir.parent.parent / "module_08_frameworks" / ".env"

if (_project_dir / ".env").exists():
    load_dotenv(_project_dir / ".env")
elif _module08_env.exists():
    load_dotenv(_module08_env)
else:
    print("⚠️  No .env file found. Set OPENROUTER_API_KEY manually.")


# =============================================================================
# LLM Configuration
# =============================================================================
# DESIGN DECISION: Why free models?
# For learning and demos, free models keep costs at $0.
# For production, you'd use GPT-4o or Claude 3.5 Sonnet.

LLM_MODEL = "inclusionai/ling-3.0-flash-vl:free"
LLM_API_BASE = "https://openrouter.ai/api/v1"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_TEMPERATURE = 0  # 0 = deterministic (important for evaluation consistency)
LLM_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "Enterprise RAG Project",
}


# =============================================================================
# Embedding Configuration
# =============================================================================
# DESIGN DECISION: Why local embeddings instead of API?
# 1. FREE — no per-token cost for embeddings
# 2. FAST — no network latency (important when embedding 1000+ chunks)
# 3. PRIVACY — documents never leave your machine
# 4. REPRODUCIBLE — same model always gives same vectors
#
# Trade-off: all-MiniLM-L6-v2 (384 dims) is weaker than
# text-embedding-3-large (3072 dims) but good enough for demos.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


# =============================================================================
# Chunking Configuration
# =============================================================================
# DESIGN DECISION: Why these specific values?
# - 500 chars ≈ 100-125 tokens ≈ a good paragraph
# - 100 char overlap ensures we don't cut sentences at boundaries
# - These are STARTING points — evaluation (Phase 4) will tell us
#   if we need to adjust

CHUNK_SIZE = 500           # characters per chunk
CHUNK_OVERLAP = 100        # characters of overlap between chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # for recursive splitting


# =============================================================================
# Retrieval Configuration
# =============================================================================
# DESIGN DECISION: Why retrieve 10 then rerank to 5?
# Retrieving more candidates gives the reranker more to work with.
# The reranker (cross-encoder) is slow but accurate — so we run it
# on 10 candidates, not 1000. This is a common production pattern:
#   Fast retrieval (10-20 candidates) → Slow reranker (top 3-5)

RETRIEVAL_TOP_K = 10       # initial retrieval candidates
RERANK_TOP_K = 5           # final results after reranking
RRF_K = 60                 # RRF constant (standard value)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =============================================================================
# Vector Store Configuration
# =============================================================================
# DESIGN DECISION: Why ChromaDB?
# Local, in-process, zero setup. For production at scale:
#   - Qdrant: self-hosted, excellent filtering + payload support
#   - Pinecone: managed, auto-scaling, but vendor lock-in
#   - Weaviate: hybrid search built-in, good for multimodal

CHROMA_PERSIST_DIR = str(_project_dir / "data" / "chroma_db")
CHROMA_COLLECTION_NAME = "enterprise_rag"


# =============================================================================
# Data Configuration
# =============================================================================
SAMPLE_DOCS_DIR = str(_project_dir / "data" / "sample_docs")


# =============================================================================
# Helper: Get configured LLM instance
# =============================================================================
def get_llm():
    """Get a configured LangChain LLM instance."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_API_BASE,
        temperature=LLM_TEMPERATURE,
        default_headers=LLM_HEADERS,
    )


def get_embedding_model():
    """Get a configured HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# =============================================================================
# Print config summary (useful for debugging)
# =============================================================================
def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("  ENTERPRISE RAG SYSTEM — Configuration")
    print("=" * 60)
    print(f"  LLM:        {LLM_MODEL}")
    print(f"  API Base:   {LLM_API_BASE}")
    print(f"  API Key:    {'✅ set' if LLM_API_KEY else '❌ MISSING'}")
    print(f"  Embeddings: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS}d)")
    print(f"  Chunk Size: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    print(f"  Retrieval:  top-{RETRIEVAL_TOP_K} → rerank to top-{RERANK_TOP_K}")
    print(f"  Vector DB:  ChromaDB at {CHROMA_PERSIST_DIR}")
    print(f"  Documents:  {SAMPLE_DOCS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
