"""
Chunking Strategies — 3 Approaches Compared
=============================================

DESIGN DECISION: Why implement 3 strategies?

In interviews, you'll be asked "how do you choose a chunking strategy?"
The only honest answer is: "I experiment and measure." This module
implements 3 strategies so we can compare them quantitatively in
the evaluation phase (Phase 4).

Strategy 1: FIXED-SIZE (baseline)
  Split every N characters. Dumb but predictable.
  Pros: Simple, consistent chunk sizes
  Cons: Cuts mid-sentence, mid-paragraph, mid-table

Strategy 2: RECURSIVE CHARACTER (LangChain default)
  Try splitting on paragraphs first, then sentences, then words.
  Pros: Respects natural boundaries
  Cons: Chunk sizes vary; doesn't understand semantics

Strategy 3: SEMANTIC (embedding-based)
  Group sentences that are semantically similar.
  Pros: Chunks are topically coherent
  Cons: Slower (needs embeddings), harder to debug

The key insight: chunking quality directly impacts retrieval quality.
If a chunk contains two unrelated topics, it will match queries for
BOTH topics — adding noise to one of them.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass

# Ensure Windows stdout handles UTF-8 characters properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))



@dataclass
class Chunk:
    """A text chunk with metadata tracing it back to the source document."""
    text: str
    metadata: dict

    def __repr__(self):
        preview = self.text[:80].replace('\n', ' ')
        return f"Chunk({len(self.text)} chars, '{preview}...')"

    def to_langchain(self):
        """Convert to LangChain Document format."""
        from langchain_core.documents import Document
        return Document(page_content=self.text, metadata=self.metadata)


# =============================================================================
# Strategy 1: Fixed-Size Chunking
# =============================================================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 100,
                     source_metadata: dict = None) -> list[Chunk]:
    """
    Split text into fixed-size chunks with overlap.

    This is the BASELINE — the simplest possible approach.
    It's useful as a comparison point but rarely used in production
    because it cuts through sentences and paragraphs blindly.

    Args:
        text: The full text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between consecutive chunks
        source_metadata: Metadata from the source document
    """
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:  # skip empty chunks
            metadata = {
                **(source_metadata or {}),
                "chunk_strategy": "fixed_size",
                "chunk_index": chunk_idx,
                "chunk_start_char": start,
                "chunk_end_char": min(end, len(text)),
            }
            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            chunk_idx += 1

        start += chunk_size - overlap

    return chunks


# =============================================================================
# Strategy 2: Recursive Character Splitting
# =============================================================================

def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 100,
                    separators: list[str] = None,
                    source_metadata: dict = None) -> list[Chunk]:
    """
    Split text recursively, trying larger separators first.

    This is LangChain's default and most popular strategy.

    The idea: try to split on double-newlines (paragraphs) first.
    If a paragraph is too long, split on single newlines.
    If still too long, split on sentences (". ").
    Last resort: split on spaces or characters.

    This respects document structure better than fixed-size.

    Args:
        text: The full text to chunk
        chunk_size: Target number of characters per chunk
        overlap: Characters of overlap between consecutive chunks
        separators: List of separators to try, from most to least preferred
        source_metadata: Metadata from the source document
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_recursive(text: str, seps: list[str]) -> list[str]:
        """Recursively split text using the first applicable separator."""
        if not text:
            return []

        # If text is small enough, return as-is
        if len(text) <= chunk_size:
            return [text]

        # Try each separator
        for i, sep in enumerate(seps):
            if sep == "":
                # Last resort: character-level split
                parts = [text[j:j + chunk_size] for j in range(0, len(text), chunk_size - overlap)]
                return parts

            if sep in text:
                splits = text.split(sep)
                # Merge splits back into chunks respecting size limit
                merged = []
                current = ""

                for split in splits:
                    candidate = current + sep + split if current else split
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            merged.append(current)
                        # If this single split is too long, recurse with next separator
                        if len(split) > chunk_size:
                            sub_splits = _split_recursive(split, seps[i + 1:])
                            merged.extend(sub_splits)
                            current = ""
                        else:
                            current = split

                if current:
                    merged.append(current)

                return merged

        return [text]

    raw_chunks = _split_recursive(text, separators)

    # Add overlap between chunks
    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Add overlap from previous chunk
        if idx > 0 and overlap > 0:
            prev_text = raw_chunks[idx - 1]
            overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            chunk_text = overlap_text.strip() + " " + chunk_text

        metadata = {
            **(source_metadata or {}),
            "chunk_strategy": "recursive",
            "chunk_index": idx,
        }
        chunks.append(Chunk(text=chunk_text, metadata=metadata))

    return chunks


# =============================================================================
# Strategy 3: Semantic Chunking
# =============================================================================

def chunk_semantic(text: str, source_metadata: dict = None,
                   similarity_threshold: float = 0.5,
                   min_chunk_size: int = 100,
                   max_chunk_size: int = 1500) -> list[Chunk]:
    """
    Group sentences by semantic similarity using embeddings.

    How it works:
      1. Split text into sentences
      2. Embed each sentence
      3. Walk through sentences sequentially
      4. If the next sentence is semantically similar to the current group,
         add it to the group
      5. If it's dissimilar (topic change), start a new chunk

    This is more expensive (requires embedding each sentence) but
    produces topically coherent chunks.

    DESIGN DECISION: Why sequential grouping instead of clustering?
    Documents have a NARRATIVE — information flows in order.
    Clustering (k-means) would group similar sentences from different
    parts of the document, breaking the narrative. Sequential grouping
    respects the document's natural flow.

    Args:
        text: The full text to chunk
        source_metadata: Metadata from the source document
        similarity_threshold: Cosine similarity threshold for grouping
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
    """
    import numpy as np

    # Step 1: Split into sentences
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(
            text=text.strip(),
            metadata={**(source_metadata or {}), "chunk_strategy": "semantic", "chunk_index": 0}
        )]

    # Step 2: Get embeddings for each sentence
    from config import get_embedding_model
    embed_model = get_embedding_model()
    embeddings = embed_model.embed_documents(sentences)
    embeddings = np.array(embeddings)

    # Step 3: Sequential grouping based on similarity
    chunks = []
    current_group = [0]  # indices of sentences in current group

    for i in range(1, len(sentences)):
        # Compare current sentence with the mean embedding of the group
        group_embedding = embeddings[current_group].mean(axis=0)
        similarity = _cosine_similarity(group_embedding, embeddings[i])

        current_text = " ".join(sentences[j] for j in current_group)

        # Add to group if similar AND not exceeding max size
        if similarity >= similarity_threshold and len(current_text) + len(sentences[i]) <= max_chunk_size:
            current_group.append(i)
        else:
            # Finalize current chunk
            chunk_text = " ".join(sentences[j] for j in current_group).strip()
            if len(chunk_text) >= min_chunk_size:
                metadata = {
                    **(source_metadata or {}),
                    "chunk_strategy": "semantic",
                    "chunk_index": len(chunks),
                }
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
            elif chunks:
                # Too small — merge with previous chunk
                chunks[-1] = Chunk(
                    text=chunks[-1].text + " " + chunk_text,
                    metadata=chunks[-1].metadata,
                )

            # Start new group
            current_group = [i]

    # Don't forget the last group
    if current_group:
        chunk_text = " ".join(sentences[j] for j in current_group).strip()
        if chunk_text:
            if len(chunk_text) >= min_chunk_size:
                metadata = {
                    **(source_metadata or {}),
                    "chunk_strategy": "semantic",
                    "chunk_index": len(chunks),
                }
                chunks.append(Chunk(text=chunk_text, metadata=metadata))
            elif chunks:
                chunks[-1] = Chunk(
                    text=chunks[-1].text + " " + chunk_text,
                    metadata=chunks[-1].metadata,
                )

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on sentence boundaries (period, question mark, exclamation)
    # but not on abbreviations like "Inc." or "$4.2B"
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on double newlines (paragraph breaks)
    expanded = []
    for s in sentences:
        parts = s.split("\n\n")
        expanded.extend(parts)
    # Filter empty strings and strip
    return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 10]


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# Comparison Runner — Run all 3 strategies and compare
# =============================================================================

def compare_strategies(text: str, source_metadata: dict = None,
                       chunk_size: int = 500, overlap: int = 100) -> dict:
    """
    Run all 3 chunking strategies and return comparison data.

    This is the key function for Phase 1 testing — it lets you see
    how each strategy handles the same document differently.
    """
    print(f"\n  Chunking {len(text):,} characters of text...\n")

    results = {}

    # Strategy 1: Fixed-size
    print("  📐 Strategy 1: Fixed-size chunking...")
    fixed_chunks = chunk_fixed_size(text, chunk_size, overlap, source_metadata)
    results["fixed_size"] = fixed_chunks
    _print_strategy_stats("Fixed-Size", fixed_chunks)

    # Strategy 2: Recursive
    print("  🔄 Strategy 2: Recursive character splitting...")
    recursive_chunks = chunk_recursive(text, chunk_size, overlap,
                                       source_metadata=source_metadata)
    results["recursive"] = recursive_chunks
    _print_strategy_stats("Recursive", recursive_chunks)

    # Strategy 3: Semantic
    print("  🧠 Strategy 3: Semantic chunking...")
    semantic_chunks = chunk_semantic(text, source_metadata)
    results["semantic"] = semantic_chunks
    _print_strategy_stats("Semantic", semantic_chunks)

    return results


def _print_strategy_stats(name: str, chunks: list[Chunk]):
    """Print statistics for a chunking strategy."""
    if not chunks:
        print(f"     {name}: 0 chunks\n")
        return

    sizes = [len(c.text) for c in chunks]
    print(f"     {name}:")
    print(f"       Chunks: {len(chunks)}")
    print(f"       Avg size: {sum(sizes) // len(sizes)} chars")
    print(f"       Min/Max: {min(sizes)}/{max(sizes)} chars")
    print(f"       Sample (chunk 0): \"{chunks[0].text[:100]}...\"")
    print()


# =============================================================================
# CLI — Run this file directly to compare chunking strategies
# =============================================================================

if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents

    print("=" * 70)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 70)

    # Create sample docs if needed
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        print("\nCreating sample data first...\n")
        create_sample_documents(SAMPLE_DOCS_DIR)

    # Load documents
    print(f"\nLoading documents from: {SAMPLE_DOCS_DIR}\n")
    docs = load_documents(SAMPLE_DOCS_DIR)

    # Compare strategies on each document
    for doc in docs:
        print("\n" + "=" * 70)
        print(f"  📄 Document: {doc.metadata['source']}")
        print("=" * 70)

        results = compare_strategies(
            doc.content,
            source_metadata={"source": doc.metadata["source"]},
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

    # Show detailed comparison for the first document
    print("\n" + "=" * 70)
    print("  DETAILED CHUNK COMPARISON — First Document")
    print("=" * 70)

    if docs:
        first_doc = docs[0]
        results = compare_strategies(
            first_doc.content,
            source_metadata={"source": first_doc.metadata["source"]},
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        print("\n  --- Fixed-size chunk boundaries (first 3) ---")
        for i, chunk in enumerate(results["fixed_size"][:3]):
            print(f"\n  Chunk {i}: [{len(chunk.text)} chars]")
            print(f"  START: \"{chunk.text[:60]}...\"")
            print(f"  END:   \"...{chunk.text[-60:]}\"")

        print("\n  --- Recursive chunk boundaries (first 3) ---")
        for i, chunk in enumerate(results["recursive"][:3]):
            print(f"\n  Chunk {i}: [{len(chunk.text)} chars]")
            print(f"  START: \"{chunk.text[:60]}...\"")
            print(f"  END:   \"...{chunk.text[-60:]}\"")

        print("\n  --- Semantic chunk boundaries (first 3) ---")
        for i, chunk in enumerate(results["semantic"][:3]):
            print(f"\n  Chunk {i}: [{len(chunk.text)} chars]")
            print(f"  START: \"{chunk.text[:60]}...\"")
            print(f"  END:   \"...{chunk.text[-60:]}\"")

    print("\n✅ Chunking comparison complete!")
    print("\n💡 Notice how:")
    print("   - Fixed-size cuts through sentences and tables")
    print("   - Recursive respects paragraph boundaries")
    print("   - Semantic groups related content together")
    print("   → We'll measure which works best in Phase 4 (RAGAS evaluation)")
