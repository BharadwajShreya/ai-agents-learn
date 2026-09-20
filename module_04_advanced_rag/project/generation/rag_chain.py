"""
RAG Chain — Query → Retrieve → Generate with Citations
=========================================================

This is the HEART of the RAG system. It orchestrates:
  1. Take a user question
  2. Retrieve relevant chunks from the hybrid search engine
  3. Build a prompt with the retrieved context
  4. Call the LLM to generate an answer WITH source citations
  5. Parse the response into a structured RAGResponse

DESIGN DECISION: Why build our own chain instead of using LangChain's
RetrievalQA or create_retrieval_chain?

LangChain chains are convenient but:
  1. They hide the prompt — you can't easily see/debug what goes to the LLM
  2. They use generic prompts — not optimized for financial data + citations
  3. They couple retrieval + generation tightly — hard to swap components
  4. They make error handling opaque — "something failed in the chain"

Our approach: explicit Python functions that you can read, debug, and modify.
Every step is visible and testable independently.

PROMPT ENGINEERING FOR RAG:
  The prompt is CRITICAL. A bad prompt will:
    - Hallucinate facts not in the context
    - Ignore source citations
    - Paraphrase instead of answering precisely

  Our prompt includes:
    - Clear role instruction (financial analyst)
    - Numbered source chunks with metadata
    - Explicit citation format instructions
    - "If not found" fallback behavior
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field

# Ensure Windows stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.llm_client import LLMClient, LLMResponse
from retrieval.vector_store import SearchResult


# =============================================================================
# System Prompt — Defines HOW the LLM should behave
# =============================================================================

SYSTEM_PROMPT = """You are a precise financial analyst assistant. Your job is to answer questions using ONLY the provided source documents.

RULES:
1. Answer ONLY based on the provided context. Do NOT use your general knowledge.
2. Cite your sources using [Source N] notation (e.g., [Source 1], [Source 2]).
3. If the context doesn't contain enough information to answer, say "I cannot find this information in the provided documents."
4. Be specific — include exact numbers, percentages, and dollar amounts when available.
5. If multiple sources provide the same information, cite all of them.
6. Keep your answer concise but complete."""


# =============================================================================
# User Prompt Template — Builds the actual prompt sent to the LLM
# =============================================================================

def build_rag_prompt(query: str, search_results: list[SearchResult]) -> str:
    """
    Build the RAG prompt with retrieved context.

    DESIGN DECISION: Why numbered sources?

    We number each source chunk [Source 1], [Source 2], etc. so that:
      1. The LLM can cite specific sources in its answer
      2. The user can verify claims against the original documents
      3. We can programmatically check if citations are valid

    We also include metadata (filename, chunk strategy) so the LLM
    knows WHERE the information came from.

    Args:
        query: The user's question
        search_results: Retrieved chunks from the hybrid search engine

    Returns:
        The complete prompt string to send to the LLM
    """
    # Build the context section with numbered sources
    context_parts = []
    for i, result in enumerate(search_results, start=1):
        source = result.metadata.get("source", "unknown")
        score = result.score
        context_parts.append(
            f"[Source {i}] (from: {source}, relevance: {score:.3f})\n"
            f"{result.text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Build the full prompt
    prompt = f"""Based on the following source documents, answer the question.

SOURCE DOCUMENTS:
{context}

---

QUESTION: {query}

ANSWER (cite sources using [Source N] notation):"""

    return prompt


# =============================================================================
# RAG Response — Structured output from the RAG pipeline
# =============================================================================

@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.

    This bundles everything the UI needs to display:
      - The generated answer text
      - The source chunks that were used (for verification)
      - Token usage (for cost monitoring)
      - The retrieval strategy that was used
    """
    answer: str
    sources: list[SearchResult]
    query: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_strategy: str

    def __repr__(self):
        return (f"RAGResponse(answer_chars={len(self.answer)}, "
                f"sources={len(self.sources)}, "
                f"tokens={self.prompt_tokens + self.completion_tokens})")

    def display(self):
        """Pretty-print the RAG response for terminal output."""
        print(f"\n  {'=' * 60}")
        print(f"  ANSWER")
        print(f"  {'=' * 60}")
        print(f"  {self.answer}")

        print(f"\n  {'─' * 60}")
        print(f"  SOURCES ({len(self.sources)} chunks used)")
        print(f"  {'─' * 60}")
        for i, source in enumerate(self.sources, start=1):
            src_name = source.metadata.get("source", "?")
            preview = source.text[:100].replace('\n', ' ')
            print(f"  [Source {i}] {src_name} (score={source.score:.3f})")
            print(f"    \"{preview}...\"")

        print(f"\n  Model: {self.model}")
        print(f"  Tokens: {self.prompt_tokens + self.completion_tokens} "
              f"(prompt={self.prompt_tokens}, completion={self.completion_tokens})")
        print(f"  Strategy: {self.retrieval_strategy}")


# =============================================================================
# RAG Chain — The main pipeline class
# =============================================================================

class RAGChain:
    """
    The complete RAG pipeline: Query → Retrieve → Generate.

    This is the top-level class that the UI (Streamlit) will use.

    USAGE:
        chain = RAGChain()
        chain.index_documents(chunks)
        response = chain.ask("What was Acme's Q3 revenue?")
        response.display()
    """

    def __init__(self, retrieval_strategy: str = "hybrid"):
        """
        Initialize the RAG chain.

        Args:
            retrieval_strategy: "hybrid", "dense", or "sparse"
        """
        from retrieval.hybrid_search import HybridSearchEngine

        self._llm = LLMClient()
        self._retrieval = HybridSearchEngine()
        self._strategy = retrieval_strategy

    def index_documents(self, chunks: list) -> dict:
        """Index chunks into the retrieval engine."""
        return self._retrieval.index(chunks)

    def ask(self, query: str, top_k: int = 5,
            strategy: str = None) -> RAGResponse:
        """
        Ask a question and get a RAG-generated answer with citations.

        THE RAG PIPELINE:
          1. RETRIEVE: Get top_k relevant chunks
          2. BUILD PROMPT: Insert chunks into the prompt template
          3. GENERATE: Send prompt to LLM
          4. PACKAGE: Return structured RAGResponse

        Args:
            query: The user's question
            top_k: Number of source chunks to retrieve
            strategy: Override retrieval strategy for this query

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        use_strategy = strategy or self._strategy

        # Step 1: RETRIEVE
        search_results = self._retrieval.search(
            query,
            top_k=top_k,
            strategy=use_strategy,
            use_reranker=(use_strategy == "hybrid"),
        )

        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer this question.",
                sources=[],
                query=query,
                model="none",
                prompt_tokens=0,
                completion_tokens=0,
                retrieval_strategy=use_strategy,
            )

        # Step 2: BUILD PROMPT
        prompt = build_rag_prompt(query, search_results)

        # Step 3: GENERATE
        llm_response = self._llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Step 4: PACKAGE
        return RAGResponse(
            answer=llm_response.content,
            sources=search_results,
            query=query,
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            retrieval_strategy=use_strategy,
        )


# =============================================================================
# CLI — Run the full RAG pipeline
# =============================================================================
if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    print("=" * 70)
    print("  RAG CHAIN — Full Pipeline Test")
    print("=" * 70)

    # Step 1: Load and chunk
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        create_sample_documents(SAMPLE_DOCS_DIR)

    print(f"\n  Loading documents...\n")
    docs = load_documents(SAMPLE_DOCS_DIR)

    all_chunks = []
    for doc in docs:
        chunks = chunk_recursive(
            doc.content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            source_metadata={"source": doc.metadata["source"]},
        )
        all_chunks.extend(chunks)

    # Step 2: Initialize RAG chain and index
    print(f"\n  Initializing RAG chain...")
    chain = RAGChain(retrieval_strategy="hybrid")

    # Only re-index if empty
    if chain._retrieval._dense.count == 0:
        chain.index_documents(all_chunks)
    else:
        # Still need to build BM25 (in-memory, lost on restart)
        chain._retrieval._sparse.index_chunks(all_chunks)
        print(f"  ChromaDB already has {chain._retrieval._dense.count} chunks")
        print(f"  Rebuilt BM25 index: {chain._retrieval._sparse.count} chunks")

    # Step 3: Ask questions!
    test_questions = [
        "What was Acme Corporation's cloud services revenue in Q3 2025, and how much did it grow?",
        "How much did Acme pay for the NeuralPath acquisition, and what did they get?",
        "Compare Acme and TechGiant's market share in enterprise cloud.",
        "What are Acme's main chip suppliers and what percentage does each provide?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print(f"  QUESTION: {question}")
        print("=" * 70)

        response = chain.ask(question)
        response.display()

    print("\n" + "=" * 70)
    print("  ✅ RAG Chain test complete!")
    print("=" * 70)
