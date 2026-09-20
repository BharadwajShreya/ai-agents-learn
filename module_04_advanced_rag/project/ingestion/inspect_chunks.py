"""
Inspect Chunks — See the actual text output of each chunking strategy.

Usage: python -m ingestion.inspect_chunks

This prints every chunk (numbered) for one document so you can
visually compare how each strategy carves up the same text.
"""

import sys
from pathlib import Path

# Ensure Windows stdout handles UTF-8 characters properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.document_loader import load_documents
from ingestion.sample_data import create_sample_documents
from ingestion.chunking import chunk_fixed_size, chunk_recursive, chunk_semantic


def print_chunks(strategy_name: str, chunks: list, max_display: int = 5):
    """Print chunks with clear visual boundaries."""
    print(f"\n{'=' * 70}")
    print(f"  {strategy_name}  —  {len(chunks)} chunks total")
    print(f"{'=' * 70}")

    for i, chunk in enumerate(chunks[:max_display]):
        print(f"\n  ┌─── Chunk {i} [{len(chunk.text)} chars] ───")
        # Show full text but indent it
        for line in chunk.text.split("\n"):
            print(f"  │ {line}")
        print(f"  └{'─' * 50}")

    if len(chunks) > max_display:
        print(f"\n  ... ({len(chunks) - max_display} more chunks not shown)")
        print(f"  ... Set max_display={len(chunks)} to see all")


if __name__ == "__main__":
    # Create sample docs if needed
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        create_sample_documents(SAMPLE_DOCS_DIR)

    # Load documents
    print("Loading documents...\n")
    docs = load_documents(SAMPLE_DOCS_DIR)

    # Pick the first markdown doc (Acme Q2 — smaller, easier to read)
    doc = docs[0]
    print(f"\nInspecting: {doc.metadata['source']} ({len(doc.content):,} chars)")

    source_meta = {"source": doc.metadata["source"]}

    # Strategy 1: Fixed-size
    fixed = chunk_fixed_size(doc.content, CHUNK_SIZE, CHUNK_OVERLAP, source_meta)
    print_chunks("STRATEGY 1: Fixed-Size (500 chars, 100 overlap)", fixed)

    # Strategy 2: Recursive
    recursive = chunk_recursive(doc.content, CHUNK_SIZE, CHUNK_OVERLAP,
                                source_metadata=source_meta)
    print_chunks("STRATEGY 2: Recursive Character Splitting", recursive)

    # Strategy 3: Semantic
    print("\n  (Computing sentence embeddings for semantic chunking...)")
    semantic = chunk_semantic(doc.content, source_meta)
    print_chunks("STRATEGY 3: Semantic Chunking (cosine similarity)", semantic)

    print(f"\n{'=' * 70}")
    print("  KEY OBSERVATIONS")
    print(f"{'=' * 70}")
    print("""
  Look at where each strategy CUTS the text:

  1. FIXED-SIZE: Cuts at exactly 500 chars regardless of content.
     → You'll see sentences sliced mid-word.

  2. RECURSIVE: Cuts at paragraph (\\n\\n) or sentence (". ") boundaries.
     → Sentences stay intact, but unrelated paragraphs may be merged.

  3. SEMANTIC: Cuts where the TOPIC changes (cosine similarity drops).
     → Each chunk is about ONE coherent topic.
     → But chunk sizes vary widely (some very small, some large).

  In Phase 4 (RAGAS evaluation), we'll measure which strategy produces
  the best retrieval accuracy for our specific financial corpus.
""")
