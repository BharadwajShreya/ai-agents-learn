"""
Module 4 — Session 4.1: Chunking Strategies Deep Dive
=====================================================
Compare 4 chunking strategies on the SAME document.
Run this to SEE how each strategy breaks text differently.
"""

# ============================================================
# SAMPLE DOCUMENT — A mini financial report with structure
# ============================================================
SAMPLE_DOCUMENT = """
# Acme Corp — Q3 2025 Earnings Report

## Executive Summary

Acme Corp reported strong Q3 results, with revenue reaching $4.2 billion,
a 15% increase year-over-year. Net income was $890 million, driven by
growth in the cloud services division. The company raised its full-year
guidance to $16.5 billion in revenue.

## Revenue Breakdown

### Cloud Services
Cloud revenue grew 28% YoY to $2.1 billion. This was driven by
enterprise adoption of our AI platform, which now serves 3,400+ customers.
Average contract value increased to $620K from $480K in Q3 2024.

### Hardware Division
Hardware revenue declined 8% to $1.4 billion. The decline was expected
as the company transitions from legacy server products to next-generation
AI accelerator chips. New chip orders increased 45% but won't ship
until Q1 2026.

### Professional Services
Professional services revenue was $700 million, flat year-over-year.
Margins improved from 22% to 27% due to automation of delivery processes.

## Key Metrics

| Metric | Q3 2025 | Q3 2024 | Change |
|--------|---------|---------|--------|
| Revenue | $4.2B | $3.65B | +15% |
| Net Income | $890M | $720M | +24% |
| Cloud ARR | $8.4B | $6.6B | +27% |
| Customers | 3,400+ | 2,800+ | +21% |
| Employees | 45,200 | 42,100 | +7% |

## Risk Factors

The company faces several risks including: increasing competition in
the cloud AI market from both established players and startups,
potential supply chain disruptions for AI chips, and regulatory
uncertainty around AI governance in the EU and US markets. Additionally,
the transition from hardware to cloud services may temporarily impact
margins as the company invests in data center infrastructure.

## Forward Guidance

For Q4 2025, the company expects revenue of $4.4-4.6 billion and
net income of $920-980 million. Full-year 2025 guidance was raised
to $16.5 billion from the previous estimate of $15.8 billion.
The company plans to invest $2 billion in AI R&D in 2026.
""".strip()

print("=" * 70)
print("DOCUMENT LENGTH")
print("=" * 70)
print(f"Characters: {len(SAMPLE_DOCUMENT)}")
print(f"Words:      {len(SAMPLE_DOCUMENT.split())}")
print(f"Lines:      {len(SAMPLE_DOCUMENT.splitlines())}")
print(f"Paragraphs: {len([p for p in SAMPLE_DOCUMENT.split(chr(10)*2) if p.strip()])}")

# ============================================================
# STRATEGY 1: Fixed-Size Chunking
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 1: FIXED-SIZE CHUNKING")
print("=" * 70)
print("Rule: Split every N characters, with overlap.")
print("Pro:  Simple, predictable chunk sizes.")
print("Con:  BREAKS mid-sentence and mid-section!\n")

CHUNK_SIZE = 300
OVERLAP = 50

chunks_fixed = []
start = 0
while start < len(SAMPLE_DOCUMENT):
    end = start + CHUNK_SIZE
    chunk = SAMPLE_DOCUMENT[start:end]
    chunks_fixed.append(chunk)
    start = end - OVERLAP  # overlap

for i, chunk in enumerate(chunks_fixed):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    # Show first and last 80 chars to see where it cuts
    preview = chunk[:80].replace('\n', '\\n')
    ending = chunk[-80:].replace('\n', '\\n')
    print(f"  START: \"{preview}...\"")
    print(f"  END:   \"...{ending}\"")
    print()

print(f"Total chunks: {len(chunks_fixed)}")
print("⚠️  Notice how chunks break MID-SENTENCE and MID-TABLE!")

# ============================================================
# STRATEGY 2: Recursive Character Splitting
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 2: RECURSIVE CHARACTER SPLITTING")
print("=" * 70)
print("Rule: Try to split on \\n\\n first, then \\n, then sentence, then word.")
print("Pro:  Respects paragraph boundaries when possible.")
print("Con:  Still doesn't understand document structure.\n")

def recursive_split(text, chunk_size=400, separators=None):
    """
    Split text recursively using a hierarchy of separators.
    This is what LangChain's RecursiveCharacterTextSplitter does.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    
    chunks = []
    
    # If text fits in one chunk, return it
    if len(text) <= chunk_size:
        return [text]
    
    # Try each separator in order (most preferred first)
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # If a single part is too long, try next separator
                    if len(part) > chunk_size and i + 1 < len(separators):
                        sub_chunks = recursive_split(part, chunk_size, separators[i+1:])
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
    
    # Fallback: just split at chunk_size
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size].strip())
    return chunks

chunks_recursive = recursive_split(SAMPLE_DOCUMENT, chunk_size=400)

for i, chunk in enumerate(chunks_recursive):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    first_line = chunk.split('\n')[0][:80]
    last_line = [l for l in chunk.split('\n') if l.strip()][-1][:80]
    print(f"  STARTS WITH: \"{first_line}\"")
    print(f"  ENDS WITH:   \"{last_line}\"")
    print()

print(f"Total chunks: {len(chunks_recursive)}")
print("✅ Better! Splits on paragraph boundaries. But still ignores section headers.")

# ============================================================
# STRATEGY 3: Document-Structure Aware Chunking
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 3: DOCUMENT-STRUCTURE AWARE CHUNKING")
print("=" * 70)
print("Rule: Parse document structure (headers, sections), chunk by section.")
print("Pro:  Each chunk is a coherent section with its header context.")
print("Con:  Requires parsing; sections can vary wildly in size.\n")

import re

def structure_aware_split(text):
    """
    Split by markdown headers, keeping header context with each chunk.
    This is what production RAG systems should use for structured docs.
    """
    chunks = []
    # Track header hierarchy
    current_headers = {1: "", 2: "", 3: ""}
    current_content = ""
    
    for line in text.split('\n'):
        # Detect headers
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if header_match:
            # Save previous chunk
            if current_content.strip():
                # Build header breadcrumb
                breadcrumb_parts = []
                for level in [1, 2, 3]:
                    if current_headers[level]:
                        breadcrumb_parts.append(current_headers[level])
                breadcrumb = " > ".join(breadcrumb_parts)
                chunks.append({
                    "header": breadcrumb,
                    "content": current_content.strip()
                })
            
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_headers[level] = title
            # Clear lower-level headers
            for l in range(level + 1, 4):
                current_headers[l] = ""
            current_content = ""
        else:
            current_content += line + "\n"
    
    # Don't forget the last chunk
    if current_content.strip():
        breadcrumb_parts = []
        for level in [1, 2, 3]:
            if current_headers[level]:
                breadcrumb_parts.append(current_headers[level])
        breadcrumb = " > ".join(breadcrumb_parts)
        chunks.append({
            "header": breadcrumb,
            "content": current_content.strip()
        })
    
    return chunks

chunks_structure = structure_aware_split(SAMPLE_DOCUMENT)

for i, chunk in enumerate(chunks_structure):
    print(f"--- Chunk {i+1} ---")
    print(f"  HEADER:  {chunk['header']}")
    print(f"  LENGTH:  {len(chunk['content'])} chars")
    content_preview = chunk['content'][:120].replace('\n', ' ')
    print(f"  PREVIEW: \"{content_preview}...\"")
    print()

print(f"Total chunks: {len(chunks_structure)}")
print("✅ Each chunk is a coherent section with navigable header context!")
print("🎯 The header breadcrumb (e.g., 'Revenue Breakdown > Cloud Services')")
print("   gives the LLM crucial context about WHAT this chunk is about.")

# ============================================================
# STRATEGY 4: Parent-Child Chunking
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY 4: PARENT-CHILD CHUNKING")
print("=" * 70)
print("Rule: Create SMALL chunks for retrieval, but return the PARENT chunk.")
print("Pro:  Precise matching + rich context. Best of both worlds!")
print("Con:  More complex, needs parent-child relationship tracking.\n")

def parent_child_split(text, child_size=150):
    """
    Create small 'child' chunks for precise vector search,
    but map each child back to its 'parent' section for context.
    
    This is the KEY insight:
    - SMALL chunks → better embedding match (more specific)
    - LARGE context → better LLM generation (more information)
    """
    # Parents are the structure-aware chunks
    parents = structure_aware_split(text)
    
    all_children = []
    for parent_idx, parent in enumerate(parents):
        # Split parent content into small children
        sentences = re.split(r'(?<=[.!?])\s+', parent['content'])
        child_text = ""
        child_idx = 0
        for sentence in sentences:
            if len(child_text) + len(sentence) > child_size and child_text:
                all_children.append({
                    "child_text": child_text.strip(),
                    "parent_idx": parent_idx,
                    "parent_header": parent['header'],
                    "parent_content": parent['content']
                })
                child_text = sentence
                child_idx += 1
            else:
                child_text += " " + sentence if child_text else sentence
        if child_text.strip():
            all_children.append({
                "child_text": child_text.strip(),
                "parent_idx": parent_idx,
                "parent_header": parent['header'],
                "parent_content": parent['content']
            })
    
    return all_children

children = parent_child_split(SAMPLE_DOCUMENT, child_size=150)

print("CHILD CHUNKS (what gets embedded for search):")
print("-" * 50)
for i, child in enumerate(children):
    print(f"  Child {i+1}: \"{child['child_text'][:80]}...\"")
    print(f"           ↳ Parent: [{child['parent_header']}] ({len(child['parent_content'])} chars)")
    print()

print(f"Total children: {len(children)}")
print()
print("HOW IT WORKS AT QUERY TIME:")
print("-" * 50)
print("""
  User asks: "What happened to cloud revenue?"

  Step 1: SEARCH — Match query against SMALL child chunks
          → Best match: "Cloud revenue grew 28% YoY to $2.1 billion"
  
  Step 2: EXPAND — Retrieve the PARENT of the matched child
          → Returns the ENTIRE "Cloud Services" section
  
  Step 3: GENERATE — Send the full parent chunk to the LLM
          → LLM has full context about cloud services to answer well
""")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: ALL 4 STRATEGIES")
print("=" * 70)
print(f"""
┌─────────────────────┬────────┬────────────┬───────────┬──────────────┐
│ Strategy            │ Chunks │ Avg Size   │ Coherent? │ Best For     │
├─────────────────────┼────────┼────────────┼───────────┼──────────────┤
│ 1. Fixed-Size       │ {len(chunks_fixed):>5}  │ {CHUNK_SIZE:>5} chars │ ❌ No     │ Quick & dirty│
│ 2. Recursive Split  │ {len(chunks_recursive):>5}  │ ~400 chars │ 🟡 Mostly │ General text │
│ 3. Structure-Aware  │ {len(chunks_structure):>5}  │ Variable   │ ✅ Yes    │ Structured   │
│ 4. Parent-Child     │ {len(children):>5}  │ ~150 chars │ ✅ Yes    │ Production   │
└─────────────────────┴────────┴────────────┴───────────┴──────────────┘
""")

print("🎯 INTERVIEW KEY POINTS:")
print("-" * 50)
print("""
1. CHUNK SIZE TRADE-OFF:
   Smaller chunks → More precise retrieval, but less context
   Larger chunks  → More context, but more noise in retrieval

2. THE GOLDEN RULE:
   "A chunk should contain exactly ONE idea, and enough context
   for the LLM to understand that idea without other chunks."

3. PRODUCTION RECOMMENDATION:
   Use Parent-Child or Structure-Aware chunking.
   Fixed-size is only for prototyping.

4. ALWAYS TEST:
   The best chunking strategy depends on YOUR data.
   That's why you benchmark — there's no universal best.
""")
