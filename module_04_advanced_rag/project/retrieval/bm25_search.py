"""
BM25 Sparse Search — Keyword-Based Retrieval
==============================================

DESIGN DECISION: Why do we need BM25 when we already have vector search?

Dense (vector) search finds SEMANTICALLY similar content:
  Query: "company earnings"  →  finds chunks about "revenue" and "profit"
  ✅ Great for paraphrasing and conceptual similarity

But dense search FAILS at exact matches:
  Query: "EBITDA margin Q3 2025"  →  might rank a chunk about "profit margins"
      above the chunk that literally says "EBITDA margin was 28.6%"
  Query: "NeuralPath acquisition"  →  the word "NeuralPath" is rare, so the
      embedding model may not have learned a good representation for it

BM25 (Best Match 25) is a SPARSE retrieval method:
  - Treats each word as a dimension (bag-of-words)
  - Scores documents by how many query terms they contain
  - Boosts rare terms (IDF) and penalizes very long documents

WHY "SPARSE"?
  A vocabulary of 50,000 words means 50,000-dimensional vectors.
  But each document only contains ~100 unique words → 49,900 dimensions are 0.
  That's a "sparse" vector (mostly zeros).

THE BM25 FORMULA:
  For each query term t in document d:
    score(t, d) = IDF(t) × [ f(t,d) × (k₁ + 1) ] / [ f(t,d) + k₁ × (1 - b + b × |d|/avgdl) ]

  Where:
    IDF(t) = inverse document frequency (rare words score higher)
    f(t,d) = how many times term t appears in document d
    |d|    = document length
    avgdl  = average document length across all documents
    k₁     = term frequency saturation (default 1.5)
    b      = length normalization (default 0.75)

ARCHITECTURE:
  ┌──────────────────────────────────────────────────────┐
  │                   BM25Search                          │
  │                                                       │
  │  index_chunks(chunks)                                 │
  │    → tokenize each chunk (lowercase + split on space) │
  │    → build BM25 index (IDF table + term frequencies)  │
  │                                                       │
  │  search(query, top_k=10)                              │
  │    → tokenize query                                   │
  │    → score every chunk using BM25 formula              │
  │    → return top_k ranked results                      │
  └──────────────────────────────────────────────────────┘
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass

# Ensure Windows stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.vector_store import SearchResult  # reuse the same result type


class BM25Search:
    """
    BM25 sparse keyword search over text chunks.

    DESIGN DECISION: Why not Elasticsearch?

    Elasticsearch is the production-standard for BM25, but it requires
    a running server (Java-based, ~1GB RAM minimum). For learning and
    demos, the `rank_bm25` Python package gives us the same algorithm
    in pure Python with zero infrastructure.

    For production:
      - <100K docs: rank_bm25 in-memory (what we use)
      - 100K-10M docs: Elasticsearch or OpenSearch
      - >10M docs: Elasticsearch cluster with sharding
    """

    def __init__(self):
        """Initialize an empty BM25 index."""
        self._index = None       # the BM25Okapi object
        self._chunks = []        # original Chunk objects (for text + metadata)
        self._tokenized = []     # tokenized versions of each chunk

    @property
    def count(self) -> int:
        """How many chunks are indexed."""
        return len(self._chunks)

    def index_chunks(self, chunks: list) -> int:
        """
        Build the BM25 index from chunks.

        Unlike ChromaDB (which persists to disk), BM25 is in-memory.
        This means:
          - Lightning fast (no disk I/O)
          - Must rebuild on restart (we rebuild from ChromaDB's stored chunks)
          - Memory scales with corpus size (~1KB per chunk)

        Args:
            chunks: List of Chunk objects

        Returns:
            Number of chunks indexed
        """
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)

        # Tokenize each chunk for BM25
        # BM25 works on tokens (words), not raw text.
        # Our tokenizer: lowercase → split on non-alphanumeric → filter short
        self._tokenized = [self._tokenize(c.text) for c in self._chunks]

        # Build the BM25 index
        # BM25Okapi computes IDF for every unique token across all documents
        # and stores term frequencies per document.
        self._index = BM25Okapi(self._tokenized)

        print(f"    BM25 index built: {len(self._chunks)} chunks, "
              f"{len(set(t for doc in self._tokenized for t in doc))} unique tokens")

        return len(self._chunks)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search using BM25 scoring.

        HOW IT WORKS:
          1. Tokenize the query the same way we tokenized documents
          2. For each document, compute BM25 score across all query tokens
          3. Sort by score descending
          4. Return top_k results

        BM25 SCORING INTUITION:
          - "What was Acme revenue?" → tokens: ["what", "was", "acme", "revenue"]
          - "what" and "was" appear in almost every chunk → low IDF → low score
          - "acme" appears in some chunks → medium IDF → medium score
          - "revenue" appears in few chunks → higher IDF → higher score
          - A chunk containing both "acme" AND "revenue" scores highest

        Args:
            query: The search query string
            top_k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by BM25 score (best first)
        """
        if self._index is None:
            raise ValueError("Index not built. Call index_chunks() first.")

        # Tokenize query with the same tokenizer
        query_tokens = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self._index.get_scores(query_tokens)

        # Get top_k indices sorted by score (descending)
        # argsort returns ascending, so we negate to sort descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include chunks with non-zero relevance
                results.append(SearchResult(
                    text=self._chunks[idx].text,
                    metadata=self._chunks[idx].metadata,
                    score=float(scores[idx]),
                ))

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple tokenizer: lowercase, split on non-alphanumeric, filter short.

        DESIGN DECISION: Why not use NLTK or spaCy?

        For BM25, a simple regex tokenizer works well because:
          - BM25 is inherently robust to simple tokenization
          - NLTK/spaCy add heavy dependencies (~100MB+)
          - In production, you'd use the search engine's built-in
            analyzer (Elasticsearch has analyzers with stemming, etc.)

        For better results, you could add:
          - Stemming (running → run, revenues → revenu)
          - Stop word removal (the, is, at)
          - N-grams (AI accelerator → "ai_accelerator")
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter very short tokens (1 char) — they add noise
        return [t for t in tokens if len(t) > 1]


# =============================================================================
# CLI — Run this file directly to test BM25 search
# =============================================================================
if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    print("=" * 70)
    print("  BM25 SPARSE SEARCH — Test")
    print("=" * 70)

    # Step 1: Load and chunk documents
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
    print(f"\n  Total chunks: {len(all_chunks)}")

    # Step 2: Build BM25 index
    print("\n  Building BM25 index...")
    bm25 = BM25Search()
    bm25.index_chunks(all_chunks)

    # Step 3: Test queries — notice how BM25 excels at exact term matches
    test_queries = [
        # BM25 should excel here — exact entity name
        "NeuralPath acquisition",
        # BM25 should excel here — specific financial metric
        "EBITDA margin Q3 2025",
        # Dense search usually wins here — conceptual query
        "How is the company doing financially?",
        # BM25 wins — specific entity + number
        "TechGiant market share enterprise cloud",
    ]

    print("\n" + "=" * 70)
    print("  BM25 SEARCH RESULTS")
    print("=" * 70)

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'─' * 60}")
        results = bm25.search(query, top_k=3)
        if not results:
            print("    (no results with BM25 score > 0)")
        for i, result in enumerate(results):
            print(f"    #{i+1} [BM25={result.score:.3f}] "
                  f"source={result.metadata.get('source', '?')}")
            preview = result.text[:120].replace('\n', ' ')
            print(f"       \"{preview}...\"")
        print()

    print("✅ BM25 search test complete!")
    print("\n💡 Compare these results with dense search (vector_store.py)")
    print("   to see where each method excels.")
