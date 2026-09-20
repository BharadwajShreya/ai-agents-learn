# 🤖 AI Agents & GenAI — Interview Prep

A structured learning repo covering **LLMs, RAG, AI Agents, and Frameworks** — built for Staff/Principal Engineer interview prep.

## 📁 Structure

```
ai-agents-learn/
├── module_01_llm_internals/       # Transformers, attention, training
├── module_02_reasoning_multimodal/ # Reasoning models, multimodal AI
├── module_03_prompting/           # Advanced prompting, output engineering
├── module_04_advanced_rag/        # RAG architectures + Enterprise RAG project
│   └── project/                   # 🔧 Enterprise Multimodal RAG System (in progress)
├── module_07_agents/              # Agent foundations, ReAct from scratch
├── module_08_frameworks/          # LangGraph, LlamaIndex, OpenAI SDK
│   └── 01_langgraph/
├── implementation_plan.md         # Overall learning plan & principles
├── task_tracker.md                # Progress checklist across all modules
├── learning_plan.md               # Detailed session-by-session plan
└── later.txt                      # Topics to revisit later
```

## 🚀 New System Setup

### 1. Clone the repo
```bash
git clone <repo-url>
cd ai-agents-learn
```

### 2. Module 4 — Enterprise RAG Project
```bash
cd module_04_advanced_rag/project

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Module 8 — LangGraph Agents
```bash
cd module_08_frameworks/01_langgraph

# Uses same OpenRouter API key
# Set OPENROUTER_API_KEY in your shell or a local .env file
```

## 🔑 API Keys Needed

| Key | Where to get | Used in |
|-----|-------------|---------|
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) | Module 4 RAG, Module 8 LangGraph |

## 📊 Progress

See [task_tracker.md](./task_tracker.md) for detailed progress.

| Module | Status |
|--------|--------|
| 1 — LLM Internals | 🟡 Theory done, project pending |
| 2 — Reasoning & Multimodal | 🟡 Theory done, project pending |
| 3 — Advanced Prompting | 🟡 Theory done, project pending |
| 4 — Advanced RAG | 🔵 Project in progress |
| 5 — Fine-Tuning | ⬜ Not started |
| 6 — SLMs & Edge AI | ⬜ Not started |
| 7 — Agent Foundations | 🟡 Theory + ReAct done |
| 8 — Agent Frameworks | 🟡 LangGraph done, others pending |
| 9 — Multi-Agent & Production | ⬜ Not started |
| 10 — System Design | ⬜ Not started |

## 🏗️ Architecture Principle (Module 4)

> **Pure Python first.** Core domain logic uses `@dataclass`, native APIs, and standard library.
> LangChain/LlamaIndex are used only at the **edges** (adapter pattern).
> This mirrors how Staff/Principal engineers build production ML systems.
