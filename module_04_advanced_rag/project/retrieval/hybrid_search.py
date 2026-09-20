"""
Hybrid Search — Dense + Sparse Fusion with Cross-Encoder Reranking
====================================================================

This is the CORE retrieval engine that combines everything:

  Dense Search (ChromaDB)     →  "Semantically similar"
  + Sparse Search (BM25)      →  "Exact keyword match"
  = Hybrid Candidates         →  Best of both worlds
  → Cross-Encoder Reranker    →  Precise relevance scoring

WHY HYBRID?
  Neither dense nor sparse search is universally better:

  ┌────────────────────┬──────────────────────────┬─────────────────────────┐
  │ Query Type         │ Dense Wins               │ Sparse (BM25) Wins      │
  ├────────────────────┼──────────────────────────┼─────────────────────────┤
  │ Conceptual         │ "How is the company      │                         │
  │                    │  performing?"             │                         │
  │ Paraphrase         │ "earnings growth" →      │                         │
  │                    │  finds "revenue increase" │                         │
  │ Exact entity       │                          │ "NeuralPath acquisition"│
  │ Specific number    │                          │ "EBITDA margin 28.6%"   │
  │ Rare terminology   │                          │ "HNSW index"            │
  └────────────────────┴──────────────────────────┴─────────────────────────┘

  Hybrid search: get top results from BOTH, merge, and let a smart
  reranker decide the final ordering.

RECIPROCAL RANK FUSION (RRF):
  The standard algorithm for merging two ranked lists.

  For each document d that appears in any ranked list:
    RRF_score(d) = Σ  1 / (k + rank_m(d))
                   m∈{dense, sparse}

  Where:
    k = 60 (standard constant — dampens the effect of high-ranking items)
    rank_m(d) = position of document d in ranker m's list (1-indexed)

  Example:
    Document X is rank 1 in dense, rank 5 in sparse:
      RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

    Document Y is rank 3 in dense, rank 1 in sparse:
      RRF = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323

    Document Y wins! Even though it was rank 3 in dense, its #1 in
    sparse boosts it above Document X.

CROSS-ENCODER RERANKING:
  After RRF gives us ~10 merged candidates, we run a Cross-Encoder
  that scores (query, passage) pairs directly.

  Bi-encoder (used for indexing):
    query → [encoder] → q_vec
    passage → [encoder] → p_vec
    score = cosine(q_vec, p_vec)
    ⚡ Fast (encode once, compare many) but less accurate

  Cross-encoder (used for reranking):
    (query, passage) → [encoder] → score
    🎯 Sees both texts TOGETHER (cross-attention) so it's more accurate
    🐌 Slow (must run for each pair) — that's why we only use it on 10 candidates
"""

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

from retrieval.vector_store import VectorStore, SearchResult
from retrieval.bm25_search import BM25Search


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    This is the STANDARD algorithm used by:
      - Elasticsearch (as of 8.x)
      - Azure AI Search
      - Weaviate's hybrid search
      - Most production RAG systems

    Args:
        dense_results: Ranked results from vector (dense) search
        sparse_results: Ranked results from BM25 (sparse) search
        k: RRF constant (default 60 — standard value from the original paper)

    Returns:
        Merged and re-ranked list of SearchResult objects
    """
    # Use chunk text as a unique key to identify the same chunk
    # across dense and sparse results.
    # (In production, you'd use document IDs instead.)
    rrf_scores = {}     # text → cumulative RRF score
    result_map = {}     # text → SearchResult object (keep the best metadata)

    # Score dense results
    for rank, result in enumerate(dense_results, start=1):
        key = result.text
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        result_map[key] = result

    # Score sparse results
    for rank, result in enumerate(sparse_results, start=1):
        key = result.text
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in result_map:
            result_map[key] = result

    # Sort by RRF score (descending) and build final results
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    merged = []
    for key in sorted_keys:
        original = result_map[key]
        merged.append(SearchResult(
            text=original.text,
            metadata=original.metadata,
            score=rrf_scores[key],  # RRF score replaces original score
        ))

    return merged


class HybridSearchEngine:
    """
    The complete retrieval engine: Dense + Sparse + RRF + Reranker.

    This is the top-level class that orchestrates the full retrieval pipeline.
    In production, this would be a microservice or a module within a larger
    orchestration framework.

    USAGE:
        engine = HybridSearchEngine()
        engine.index(chunks)
        results = engine.search("What was Q3 revenue?", top_k=5)
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        """Initialize both dense and sparse search engines."""
        from config import RERANKER_MODEL, RETRIEVAL_TOP_K, RERANK_TOP_K, RRF_K

        self._dense = VectorStore(persist_dir, collection_name)
        self._sparse = BM25Search()
        self._reranker = None  # lazy-loaded (heavy model)
        self._reranker_model_name = RERANKER_MODEL
        self._retrieval_top_k = RETRIEVAL_TOP_K
        self._rerank_top_k = RERANK_TOP_K
        self._rrf_k = RRF_K

    def index(self, chunks: list) -> dict:
        """
        Index chunks into both dense (ChromaDB) and sparse (BM25) stores.

        Returns a summary dict with counts.
        """
        print("\n  Indexing into dense store (ChromaDB)...")
        dense_count = self._dense.index_chunks(chunks)

        print("\n  Indexing into sparse store (BM25)...")
        sparse_count = self._sparse.index_chunks(chunks)

        return {
            "dense_indexed": dense_count,
            "sparse_indexed": sparse_count,
        }

    def search(
        self,
        query: str,
        top_k: int = None,
        use_reranker: bool = True,
        strategy: str = "hybrid",
    ) -> list[SearchResult]:
        """
        Search with configurable strategy.

        Args:
            query: The search query
            top_k: Number of final results (default: RERANK_TOP_K from config)
            use_reranker: Whether to apply cross-encoder reranking
            strategy: One of:
                - "hybrid" (default) — Dense + Sparse + RRF + Reranker
                - "dense"  — Dense only (vector similarity)
                - "sparse" — Sparse only (BM25 keyword match)

        Returns:
            List of SearchResult objects
        """
        final_top_k = top_k or self._rerank_top_k

        if strategy == "dense":
            return self._dense.search(query, top_k=final_top_k)

        elif strategy == "sparse":
            return self._sparse.search(query, top_k=final_top_k)

        elif strategy == "hybrid":
            # Step 1: Get candidates from both engines
            dense_results = self._dense.search(query, top_k=self._retrieval_top_k)
            sparse_results = self._sparse.search(query, top_k=self._retrieval_top_k)

            # Step 2: Merge with RRF
            merged = reciprocal_rank_fusion(
                dense_results, sparse_results, k=self._rrf_k
            )

            # Step 3: Optionally rerank
            if use_reranker and merged:
                merged = self._rerank(query, merged, top_k=final_top_k)
            else:
                merged = merged[:final_top_k]

            return merged
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Use 'hybrid', 'dense', or 'sparse'.")

    def _rerank(self, query: str, candidates: list[SearchResult],
                top_k: int) -> list[SearchResult]:
        """
        Rerank candidates using a cross-encoder model.

        HOW CROSS-ENCODER RERANKING WORKS:

        A bi-encoder (used for initial retrieval) embeds query and
        passage INDEPENDENTLY:
          q_emb = encode("What was revenue?")     # 384-dim vector
          p_emb = encode("Revenue was $4.2B...")   # 384-dim vector
          score = cosine(q_emb, p_emb)

        A cross-encoder sees BOTH texts TOGETHER through cross-attention:
          score = cross_encode("What was revenue?", "Revenue was $4.2B...")
          The model can see "revenue" in the query attending to "Revenue"
          and "$4.2B" in the passage → much more accurate relevance score.

        WHY NOT USE CROSS-ENCODER FOR EVERYTHING?
          Cross-encoder must run inference for EACH (query, passage) pair.
          For 1000 chunks, that's 1000 forward passes (~10 seconds).
          For 10 candidates, it's ~0.1 seconds. That's why we use it
          only after narrowing down to ~10 candidates via fast retrieval.
        """
        if self._reranker is None:
            print(f"  Loading reranker: {self._reranker_model_name}...")
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self._reranker_model_name)
            print(f"  Reranker loaded.")

        # Create (query, passage) pairs for the cross-encoder
        pairs = [(query, result.text) for result in candidates]

        # Score all pairs
        scores = self._reranker.predict(pairs)

        # Attach scores to results and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with updated scores
        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(SearchResult(
                text=result.text,
                metadata=result.metadata,
                score=float(score),  # cross-encoder score replaces RRF score
            ))

        return reranked

    def get_stats(self) -> dict:
        """Get stats from both engines."""
        return {
            "dense": self._dense.get_stats(),
            "sparse_chunks": self._sparse.count,
            "reranker": self._reranker_model_name,
        }


# =============================================================================
# CLI — Full retrieval pipeline comparison
# =============================================================================
if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    print("=" * 70)
    print("  HYBRID SEARCH ENGINE — Full Pipeline Test")
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
    print(f"\n  Total chunks: {len(all_chunks)}")

    # Step 2: Build hybrid engine
    engine = HybridSearchEngine()

    # Clear for clean test
    if engine._dense.count > 0:
        engine._dense.delete_collection()
        engine = HybridSearchEngine()

    engine.index(all_chunks)

    # Step 3: Compare all strategies on the same queries
    test_queries = [
        # Conceptual query (dense should win)
        "How is the company doing financially?",
        # Exact entity (BM25 should win)
        "NeuralPath acquisition price",
        # Mixed query (hybrid should win)
        "What was Acme's cloud revenue growth in Q3 2025?",
        # Cross-entity comparison (needs both)
        "Compare TechGiant and Acme market share",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"  QUERY: \"{query}\"")
        print("=" * 70)

        for strategy in ["dense", "sparse", "hybrid"]:
            print(f"\n  --- {strategy.upper()} ---")
            results = engine.search(query, top_k=3, strategy=strategy,
                                    use_reranker=(strategy == "hybrid"))
            if not results:
                print("    (no results)")
                continue
            for i, r in enumerate(results):
                source = r.metadata.get("source", "?")
                preview = r.text[:100].replace('\n', ' ')
                print(f"    #{i+1} [score={r.score:.4f}] {source}")
                print(f"        \"{preview}...\"")

    print("\n" + "=" * 70)
    print("  ENGINE STATS")
    print("=" * 70)
    stats = engine.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")

    print("\n✅ Hybrid search pipeline test complete!")
