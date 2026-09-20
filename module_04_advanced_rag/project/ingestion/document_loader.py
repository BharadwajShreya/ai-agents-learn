"""
Document Loader — Load PDFs, Markdown, and Text Files
======================================================

DESIGN DECISION: Why a custom loader instead of LangChain's DirectoryLoader?

LangChain's DirectoryLoader works but:
  1. It's a black box — you don't know how it handles different file types
  2. It doesn't give you control over metadata extraction
  3. For PDFs, you want to extract images separately (for multimodal)

For an architect, understanding the ingestion pipeline internals matters
because this is where most real-world RAG failures begin:
  - Garbled PDF text → bad chunks → bad retrieval → bad answers
  - Missing metadata → can't filter by source/page → noise in results

In production, you'd use a document intelligence service (Azure DI,
AWS Textract, Google Document AI) for complex PDFs.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Ensure Windows stdout handles UTF-8 characters properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



@dataclass
class Document:
    """
    A loaded document with content and metadata.

    DESIGN DECISION: Why a custom Document class instead of LangChain's?
    We want to track additional metadata (images, tables) that
    LangChain's Document doesn't support natively. We'll convert
    to LangChain Documents when needed for the vector store.
    """
    content: str
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)  # extracted image paths

    def to_langchain(self):
        """Convert to LangChain Document format."""
        from langchain_core.documents import Document as LCDocument
        return LCDocument(page_content=self.content, metadata=self.metadata)

    def __repr__(self):
        return (
            f"Document(source={self.metadata.get('source', '?')}, "
            f"chars={len(self.content)}, images={len(self.images)})"
        )


def load_markdown(filepath: str) -> Document:
    """Load a markdown file as a Document."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    return Document(
        content=content,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "file_type": "markdown",
            "char_count": len(content),
        }
    )


def load_pdf(filepath: str) -> Document:
    """
    Load a PDF file, extracting text and images.

    DESIGN DECISION: Why PyMuPDF over PyPDF?
    PyMuPDF (fitz) is:
      - 5-10x faster than PyPDF for text extraction
      - Better at preserving text layout and ordering
      - Can extract embedded images (needed for multimodal)

    We extract text page-by-page and store page numbers in metadata.
    Images are saved to disk for later multimodal processing.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("pymupdf is required for PDF loading. Run: pip install pymupdf")

    path = Path(filepath)
    doc = fitz.open(str(path))

    pages_text = []
    images = []

    for page_num, page in enumerate(doc, start=1):
        # Extract text
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_num}]\n{text}")

        # Extract images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image:
                img_ext = base_image["ext"]
                img_bytes = base_image["image"]
                # Save image to disk
                img_dir = path.parent / "extracted_images"
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"{path.stem}_p{page_num}_img{img_idx}.{img_ext}"
                img_path.write_bytes(img_bytes)
                images.append({
                    "path": str(img_path),
                    "page": page_num,
                    "index": img_idx,
                })

    doc.close()

    full_text = "\n\n".join(pages_text)

    return Document(
        content=full_text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "file_type": "pdf",
            "num_pages": len(pages_text),
            "char_count": len(full_text),
            "num_images": len(images),
        },
        images=images,
    )


def load_text(filepath: str) -> Document:
    """Load a plain text file as a Document."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    return Document(
        content=content,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "file_type": "text",
            "char_count": len(content),
        }
    )


# File type → loader mapping
LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".txt": load_text,
}


def load_documents(directory: str) -> list[Document]:
    """
    Load all supported documents from a directory.

    Returns a list of Document objects with content and metadata.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents = []
    skipped = []

    for filepath in sorted(dir_path.iterdir()):
        if filepath.is_file():
            ext = filepath.suffix.lower()
            if ext in LOADERS:
                try:
                    doc = LOADERS[ext](str(filepath))
                    documents.append(doc)
                    print(f"  ✅ Loaded: {filepath.name} "
                          f"({doc.metadata['char_count']:,} chars"
                          f"{', ' + str(doc.metadata.get('num_images', 0)) + ' images' if doc.images else ''})")
                except Exception as e:
                    print(f"  ❌ Error loading {filepath.name}: {e}")
                    skipped.append(filepath.name)
            else:
                skipped.append(filepath.name)

    if skipped:
        print(f"\n  ⏭️  Skipped {len(skipped)} unsupported files: {', '.join(skipped[:5])}")

    print(f"\n  📄 Loaded {len(documents)} documents total")
    return documents


# =============================================================================
# CLI — Run this file directly to test document loading
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import SAMPLE_DOCS_DIR
    from ingestion.sample_data import create_sample_documents

    print("=" * 60)
    print("  DOCUMENT LOADER — Test Run")
    print("=" * 60)

    # Create sample docs if they don't exist
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        print("\nNo documents found. Creating sample data...\n")
        create_sample_documents(SAMPLE_DOCS_DIR)

    print(f"\nLoading documents from: {SAMPLE_DOCS_DIR}\n")
    docs = load_documents(SAMPLE_DOCS_DIR)

    print("\n" + "-" * 60)
    print("  Document Summary")
    print("-" * 60)
    for doc in docs:
        print(f"  {doc.metadata['source']:40} {doc.metadata['char_count']:>6,} chars  "
              f"type={doc.metadata['file_type']}")
