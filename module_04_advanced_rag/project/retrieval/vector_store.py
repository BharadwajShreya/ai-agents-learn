"""
Vector Store — ChromaDB Dense Search (Pure Python API)
=======================================================

DESIGN DECISION: Why native ChromaDB API instead of LangChain's Chroma wrapper?

LangChain's Chroma class (langchain_chroma.Chroma) wraps ChromaDB with:
  - Automatic embedding generation
  - A .as_retriever() method for chains

But it HIDES critical details:
  1. How embeddings are stored and indexed
  2. How metadata filtering works under the hood
  3. What happens when you update/delete documents
  4. Collection management (multiple collections for A/B testing)

Using ChromaDB's native API gives us:
  - Full control over collection lifecycle
  - Direct metadata filtering (e.g. "only search chunks from PDFs")
  - Batch operations for performance
  - Clear understanding of what's happening (no abstraction magic)

For interviews, being able to explain ChromaDB internals directly
(not through LangChain's lens) signals deeper understanding.

ARCHITECTURE:
  ┌──────────────────────────────────────────────────────┐
  │                   VectorStore                         │
  │                                                       │
  │  index_chunks(chunks)                                 │
  │    → embed each chunk's text using all-MiniLM-L6-v2   │
  │    → store text + embedding + metadata in ChromaDB    │
  │                                                       │
  │  search(query, top_k=10)                              │
  │    → embed the query                                  │
  │    → find top_k nearest neighbors by cosine distance  │
  │    → return ranked results with scores + metadata     │
  └──────────────────────────────────────────────────────┘
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


@dataclass
class SearchResult:
    """
    A single search result from the vector store.

    Pure Python dataclass — no framework dependency.
    Contains everything the downstream generation engine needs:
      - The actual text to feed to the LLM
      - Metadata for source citation (which file, which page)
      - A relevance score for ranking and confidence display
    """
    text: str
    metadata: dict
    score: float  # cosine similarity (higher = more relevant)

    def __repr__(self):
        preview = self.text[:80].replace('\n', ' ')
        source = self.metadata.get("source", "?")
        return f"SearchResult(score={self.score:.3f}, source={source}, '{preview}...')"


class VectorStore:
    """
    ChromaDB-backed dense vector store.

    DESIGN DECISION: Why a class instead of standalone functions?

    A class encapsulates the ChromaDB client, collection, and embedding
    model together. This means:
      1. The embedding model is loaded ONCE (not per query — saves ~2s each time)
      2. The collection reference is reused across operations
      3. Easy to swap implementations (e.g. Qdrant, Pinecone) by
         creating a new class with the same interface

    This is the Strategy Pattern — the retrieval engine doesn't care
    which vector DB is behind this interface.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        """
        Initialize the vector store.

        Args:
            persist_dir: Where ChromaDB saves its data on disk.
                         If None, uses the config default.
            collection_name: Name of the ChromaDB collection.
                             Think of it like a database table.
        """
        import chromadb
        from config import (
            CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
            EMBEDDING_MODEL, get_embedding_model
        )

        self._persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME

        # --- Initialize ChromaDB ---
        # PersistentClient saves data to disk so you don't re-index every time.
        # In production, you'd use chromadb.HttpClient() pointing to a
        # ChromaDB server running separately (containerized).
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # --- Get or Create Collection ---
        # A "collection" in ChromaDB ≈ a table in SQL.
        # cosine = most common for text similarity (normalized dot product).
        # Alternatives: "l2" (euclidean — good for spatial data),
        #               "ip" (inner product — good for recommendation).
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # --- Load Embedding Model (once) ---
        # This loads the ~80MB model into RAM. We do it once in __init__
        # so that index_chunks() and search() don't pay the load cost.
        print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
        self._embed_model = get_embedding_model()
        print(f"  Embedding model loaded.")

    @property
    def count(self) -> int:
        """How many chunks are currently indexed."""
        return self._collection.count()

    def index_chunks(self, chunks: list, batch_size: int = 50) -> int:
        """
        Embed and store chunks in ChromaDB.

        DESIGN DECISION: Why batch processing?

        Embedding 1000 chunks one-by-one makes 1000 separate calls to
        the model. Batching lets the model process multiple texts in
        one forward pass (matrix multiplication on all texts at once),
        which is 5-10x faster.

        ChromaDB also benefits from batch upserts — fewer disk writes.

        Args:
            chunks: List of Chunk objects (from chunking.py)
            batch_size: Number of chunks to embed and store at once

        Returns:
            Number of chunks indexed
        """
        total = len(chunks)
        indexed = 0

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            # Extract text and metadata from our Chunk dataclass
            texts = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]

            # Generate unique IDs for each chunk.
            # Format: "source_name::strategy::chunk_index"
            # This lets us identify and update specific chunks later.
            ids = []
            for c in batch:
                source = c.metadata.get("source", "unknown")
                strategy = c.metadata.get("chunk_strategy", "unknown")
                idx = c.metadata.get("chunk_index", 0)
                chunk_id = f"{source}::{strategy}::chunk_{idx}"
                ids.append(chunk_id)

            # Embed the batch
            # embed_documents() processes all texts in one model forward pass
            embeddings = self._embed_model.embed_documents(texts)

            # Upsert into ChromaDB
            # "upsert" = insert if new, update if ID already exists
            # This is idempotent — safe to run multiple times
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            indexed += len(batch)
            print(f"    Indexed batch {i // batch_size + 1}: "
                  f"{indexed}/{total} chunks")

        return indexed

    def search(self, query: str, top_k: int = 10,
               where: dict = None) -> list[SearchResult]:
        """
        Search for chunks most similar to the query.

        HOW DENSE SEARCH WORKS:
          1. Embed the query using the SAME model used for indexing
          2. ChromaDB uses HNSW (Hierarchical Navigable Small World) index
             to find approximate nearest neighbors in O(log n) time
          3. Returns chunks ranked by cosine similarity

        WHAT IS HNSW?
          Instead of comparing the query against ALL vectors (O(n)),
          HNSW builds a multi-layer graph where each node connects to
          its nearest neighbors. Search starts at the top layer (few nodes,
          big jumps) and drills down (many nodes, precise). This gives
          ~95% recall at 100x the speed of brute-force.

        Args:
            query: The search query string
            top_k: Number of results to return
            where: Optional metadata filter (e.g. {"source": "acme.pdf"})
                   This is ChromaDB's filtering — search ONLY within
                   documents matching the filter.

        Returns:
            List of SearchResult objects, sorted by relevance (best first)
        """
        # Embed the query
        query_embedding = self._embed_model.embed_query(query)

        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        # Execute search
        results = self._collection.query(**query_kwargs)

        # Convert ChromaDB results to our SearchResult dataclass
        # ChromaDB returns distances (lower = more similar for cosine).
        # We convert to similarity: similarity = 1 - distance
        search_results = []
        if results["documents"] and results["documents"][0]:
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance ∈ [0, 2], similarity ∈ [-1, 1]
                # For normalized vectors: similarity = 1 - distance
                similarity = 1.0 - distance
                search_results.append(SearchResult(
                    text=text,
                    metadata=metadata,
                    score=similarity,
                ))

        return search_results

    def delete_collection(self):
        """Delete the entire collection (useful for re-indexing)."""
        self._client.delete_collection(self._collection_name)
        print(f"  Deleted collection: {self._collection_name}")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "collection_name": self._collection_name,
            "total_chunks": self.count,
            "persist_dir": self._persist_dir,
        }


# =============================================================================
# CLI — Run this file directly to test vector store indexing + search
# =============================================================================
if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    print("=" * 70)
    print("  VECTOR STORE — Index & Search Test")
    print("=" * 70)

    # Step 1: Load documents
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        create_sample_documents(SAMPLE_DOCS_DIR)

    print(f"\n  Loading documents from: {SAMPLE_DOCS_DIR}\n")
    docs = load_documents(SAMPLE_DOCS_DIR)

    # Step 2: Chunk with recursive strategy (our best structural strategy)
    print("\n  Chunking documents with recursive strategy...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_recursive(
            doc.content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            source_metadata={"source": doc.metadata["source"]},
        )
        all_chunks.extend(chunks)
    print(f"  Total chunks: {len(all_chunks)}")

    # Step 3: Index into ChromaDB
    print("\n  Indexing into ChromaDB...")
    store = VectorStore()

    # Clear existing data for a clean test
    if store.count > 0:
        print(f"  (Clearing {store.count} existing chunks for clean test)")
        store.delete_collection()
        store = VectorStore()

    store.index_chunks(all_chunks)
    print(f"\n  ChromaDB now has {store.count} indexed chunks")

    # Step 4: Test searches
    test_queries = [
        "What was Acme's cloud revenue in Q3 2025?",
        "How much did AI accelerator chip revenue grow?",
        "Who are Acme's main chip suppliers?",
        "What is TechGiant's market share in enterprise cloud?",
    ]

    print("\n" + "=" * 70)
    print("  SEARCH RESULTS")
    print("=" * 70)

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'─' * 60}")
        results = store.search(query, top_k=3)
        for i, result in enumerate(results):
            print(f"    #{i+1} [score={result.score:.3f}] "
                  f"source={result.metadata.get('source', '?')}")
            preview = result.text[:120].replace('\n', ' ')
            print(f"       \"{preview}...\"")
        print()

    # Step 5: Test metadata filtering
    print("=" * 70)
    print("  FILTERED SEARCH (only PDF documents)")
    print("=" * 70)
    query = "What was the revenue?"
    results = store.search(query, top_k=3,
                           where={"source": "acme_q3_2025_earnings_report.pdf"})
    print(f"\n  Query: \"{query}\" (filtered to PDF only)")
    for i, result in enumerate(results):
        print(f"    #{i+1} [score={result.score:.3f}] "
              f"source={result.metadata.get('source', '?')}")

    print(f"\n  Stats: {store.get_stats()}")
    print("\n✅ Vector store test complete!")
