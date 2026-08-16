# Module 8: Agent Frameworks, SDKs & Orchestration

## Directory Structure

```
module_08_frameworks/
├── .env                          # Shared API keys (OpenRouter, Gemini)
├── README.md                     # This file — module overview
├── venv/                         # Virtual environment (not committed)
│
├── 01_langgraph/                 # LangGraph + LangChain deep dive
│   ├── notes.md                  # Comprehensive revision notes
│   ├── 01_langgraph_basics.py    # Core mechanics: State, Nodes, Edges (no LLM)
│   ├── 02_conditional_edges.py   # ReAct pattern as a graph (simulated)
│   ├── 03_langgraph_real_agent.py # Full agent with real LLM + tools (OpenRouter)
│   ├── 04_prebuilt_react.py      # create_react_agent() shortcut
│   └── 05_debugging_langchain.py # 4 debugging techniques for LangChain
│
├── 02_llamaindex/                # LlamaIndex — RAG-first framework
│   ├── notes.md                  # Comprehensive revision notes
│   ├── 01_core_abstractions.py   # Documents, Nodes, NodeParsers (no LLM)
│   ├── 02_rag_pipeline.py        # Full RAG pipeline with real LLM
│   ├── 03_agentic_rag.py         # Agent decides which knowledge base to query
│   ├── 04_workflows_demo.py      # Modern event-driven Workflows API
│   └── sample_data/              # Sample docs for RAG demos
│       ├── company_policies.txt
│       ├── product_docs.txt
│       └── engineering_practices.txt
│
├── 03_openai_anthropic_google/   # OpenAI SDK + Anthropic SDK + Google ADK
│   └── notes.md                  # (upcoming)
│
└── 04_mcp_a2a/                   # MCP & A2A Protocols
    └── notes.md                  # (upcoming)
```

## Session Map

| Session | Topic | Status | Folder |
|---------|-------|--------|--------|
| 8.1 | LangGraph Deep Dive | ✅ Complete | `01_langgraph/` |
| 8.2 | LlamaIndex for RAG Agents | ✅ Complete | `02_llamaindex/` |
| 8.3 | OpenAI SDK + Google ADK + Anthropic SDK | 🔲 Upcoming | `03_openai_anthropic_google/` |
| 8.4 | MCP & A2A Protocols | 🔲 Upcoming | `04_mcp_a2a/` |
| 8.5 | Multi-Tool Research Agent + Review | 🔲 Upcoming | `01_langgraph/` |

## LLM Provider Setup

All scripts use **OpenRouter** (OpenAI-compatible API) with free models.
API key is stored in `.env` at this directory level.

### Free Models Used

| Model | Where Used | Notes |
|-------|-----------|-------|
| `google/gemma-4-26b-a4b-it:free` | `03_langgraph_real_agent.py` | Good tool calling |
| `openai/gpt-oss-20b:free` | `05_debugging_langchain.py` | OpenAI's open model |
| `google/gemma-3-27b-it:free` | `02_llamaindex/*.py` | Solid general purpose |

### Embeddings

| Model | Where Used | Notes |
|-------|-----------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | `02_llamaindex/*.py` | Local HuggingFace, free |

## Virtual Environment Setup

```bash
cd module_08_frameworks
source venv/bin/activate
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-huggingface
```
