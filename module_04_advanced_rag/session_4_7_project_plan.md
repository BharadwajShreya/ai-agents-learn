# Session 4.7: Enterprise RAG System — Project Design & Build Plan

> **Status:** READY TO BUILD — All teaching sessions (4.1-4.6) complete.
> **Next step:** Start Phase 1 (Foundation — Config + Data Loading + Chunking)
> **Prerequisites:** OpenRouter API key in `.env` (already set up in module_08)

---

## 1. Requirements Analysis

### Functional Requirements

| # | Requirement | Maps To Session |
|---|------------|-----------------|
| FR1 | Ingest PDF documents (text + tables + charts) | 4.1 (Chunking), 4.5 (Multimodal) |
| FR2 | Support multiple chunking strategies with comparison | 4.1 |
| FR3 | Hybrid search (dense + sparse + reranking) | 4.3 |
| FR4 | Multimodal support (LLM describes charts/images) | 4.5 |
| FR5 | Source citations in generated answers | 4.3, 4.4 |
| FR6 | RAGAS evaluation on a test set | 4.6 |
| FR7 | Lightweight Graph RAG exercise | 4.4 |
| FR8 | Interactive chat UI with source display | — |

### Non-Functional Requirements

| # | Requirement | Design Decision |
|---|------------|-----------------|
| NFR1 | Use OpenRouter (free models) for LLM | Cost: $0 |
| NFR2 | Local embeddings (no API cost) | HuggingFace sentence-transformers |
| NFR3 | Local vector DB (no cloud dependency) | ChromaDB |
| NFR4 | Single machine, no Docker required | Simple `pip install` |
| NFR5 | Reproducible — anyone can clone and run | `.env` + `requirements.txt` |

---

## 2. Architecture Design

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE RAG SYSTEM                          │
│                                                                   │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐      │
│  │  INGESTION   │     │  RETRIEVAL    │     │  GENERATION   │     │
│  │  PIPELINE    │────▶│  ENGINE       │────▶│  ENGINE       │     │
│  └─────────────┘     └──────────────┘     └──────────────┘      │
│        │                    │                     │               │
│        ▼                    ▼                     ▼               │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐      │
│  │ ChromaDB    │     │ BM25 Index   │     │ OpenRouter    │      │
│  │ (dense)     │     │ (sparse)     │     │ (LLM)        │      │
│  └─────────────┘     └──────────────┘     └──────────────┘      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  EVALUATION ENGINE (RAGAS)                               │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  STREAMLIT CHAT UI                                       │     │
│  └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
module_04_advanced_rag/
├── project/
│   ├── .env                        # OPENROUTER_API_KEY
│   ├── requirements.txt            # All dependencies
│   ├── config.py                   # Centralized configuration
│   │
│   ├── data/                       # Source documents
│   │   └── sample_docs/            # PDFs, markdown files
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py      # PDF/text loading
│   │   ├── chunking.py             # 3 chunking strategies
│   │   └── embeddings.py           # Embedding generation
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB setup + dense search
│   │   ├── bm25_search.py          # BM25 sparse search
│   │   ├── hybrid_search.py        # RRF merge + reranker
│   │   └── graph_rag.py            # Lightweight Graph RAG
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── rag_chain.py            # Query → Retrieve → Generate
│   │   └── guardrails.py           # Hallucination checks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── test_dataset.json       # 50 Q&A pairs
│   │   ├── ragas_eval.py           # RAGAS metrics
│   │   └── compare_strategies.py   # Side-by-side retrieval comparison
│   │
│   └── app.py                      # Streamlit chat UI
```

---

## 3. Technology Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | OpenRouter → `google/gemma-3-27b-it:free` or `google/gemma-4-26b-a4b-it:free` | Free, tool support, proven in Module 8 |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, fast, ~80MB, good for demos |
| **Vector DB** | ChromaDB (local, in-process) | Zero setup, pip install |
| **Sparse search** | `rank_bm25` Python package | Pure Python BM25 |
| **Reranker** | `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local |
| **Framework** | Pure Python first; LangChain only at edges (adapter pattern) | See Architecture Principle below |
| **Evaluation** | RAGAS library | Standard for RAG evaluation |
| **UI** | Streamlit | Fast to build |
| **Multimodal** | LLM-described images (Approach 2) | Works without GPU |

### Architecture Principle: Pure Python First

> **Rule:** Use pure Python (`@dataclass`, native ChromaDB API, raw `requests`/`httpx` for LLM calls)
> for all core domain logic. Only use LangChain/LlamaIndex at the **edges** via adapter methods
> (e.g. `to_langchain()`), and only when a LangChain utility genuinely saves significant effort
> (e.g. RAGAS evaluation expects LangChain objects).
>
> **Why?**
> - **Zero framework lock-in:** If LangChain breaks or changes APIs, our ingestion/retrieval/generation code is untouched.
> - **Interview signal:** Staff/Principal architects define their own domain models; juniors couple everything to the tutorial framework.
> - **Debuggability:** When something breaks, you're debugging YOUR code with standard Python, not stepping through 12 layers of LangChain abstraction.
> - **Enterprise reality:** Production ML systems at Google, Meta, Stripe all use internal domain entities — frameworks are adapters, not the core.

**Note:** We use LLM-described images (Approach 2) instead of ColPali because ColPali requires a GPU. For interviews, explain ColPali conceptually (Session 4.5 notes) and show this working implementation as proof. The architecture is designed so ColPali could be swapped in later.

**OpenRouter Setup** (same pattern as module_08):
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Enterprise RAG Project"
    }
)
```


---

## 4. Phased Build Plan

### Phase 1: Foundation — Config + Data Loading + Chunking (~45 min)

- [ ] Project scaffold (directories, config.py, requirements.txt, .env)
- [ ] Document loader (PDF + markdown)
- [ ] 3 chunking strategies side-by-side:
  - Fixed-size (baseline)
  - Recursive character splitting
  - Semantic chunking (embedding-based boundaries)
- [ ] Script that loads docs, chunks with all 3 strategies, prints comparison

**Test:** Run script → see chunk counts, sizes, and sample chunks from each strategy.

---

### Phase 2: Embeddings + Vector Store + Retrieval (~45 min)

- [ ] Embed chunks with HuggingFace model
- [ ] Store in ChromaDB with metadata (source, page, chunk_strategy)
- [ ] Basic dense vector search
- [ ] BM25 sparse search
- [ ] Hybrid search with RRF merge
- [ ] Reranker (cross-encoder) on top results

**Test:** Run queries → see results from dense, sparse, hybrid, and reranked — compare quality.

---

### Phase 3: RAG Chain — Generation with Citations (~30 min)

- [ ] Connect retrieval → LLM generation via OpenRouter
- [ ] Prompt template with source citation instructions
- [ ] Return answer + source chunks + confidence
- [ ] Basic hallucination guardrail

**Test:** Ask questions → get answers with source citations.

---

### Phase 4: Evaluation — RAGAS Metrics (~45 min)

- [ ] Create 30-50 question test dataset (LLM-generated + reviewed)
- [ ] Run RAGAS evaluation: Faithfulness, Answer Relevance, Context Precision, Context Recall
- [ ] Compare metrics across chunking strategies
- [ ] Generate evaluation report

**Test:** Evaluation table comparing strategies with numeric metrics.

---

### Phase 5: Graph RAG Exercise (~45 min)

- [ ] LLM extracts entity-relationship triples from chunks
- [ ] Store in Python dictionary (simple graph)
- [ ] Basic graph traversal for multi-hop queries
- [ ] Compare Graph RAG vs standard RAG on relationship questions

**Test:** Side-by-side comparison showing Graph RAG answers relationship queries.

---

### Phase 6: Streamlit Chat UI (~30 min)

- [ ] Chat interface with message history
- [ ] Source document display (expandable)
- [ ] Toggle between retrieval strategies
- [ ] Show retrieval metrics per query

**Test:** Interactive chat app running locally.

---

## 5. Open Questions (to decide when starting)

1. **Sample documents:** Use financial reports, technical docs, or create a custom corpus?
2. **Multimodal scope:** Include actual PDFs with charts, or keep text-heavy?
3. **Deployment:** Deploy to HF Spaces/Streamlit Cloud for portfolio?
