# Module 4: Advanced RAG Architectures

> **Purpose:** Master production-grade RAG — from chunking to evaluation, including multimodal RAG and advanced patterns (Agentic RAG, Graph RAG, CRAG). RAG is the **#1 most asked topic** in GenAI interviews.

---

## Session 4.1: Chunking Strategies Deep Dive

### 1. THE PROBLEM — Why Do We Need RAG at All?

Modern LLMs have large context windows (GPT-4o: 128K tokens, Gemini 2.5: 1M+), so why not just put everything in the prompt?

**Three reasons it fails:**

```
Scenario: Company with 500 financial reports, ~50 pages each

1. SIZE:     500 × 50 pages × ~500 tokens/page = 12.5 MILLION tokens
             → Even Gemini's 1M context can't hold 12.5M

2. COST:     GPT-4o input: $2.50 per 1M tokens
             → $31.25 PER QUERY just on input
             → 100 queries/day = $93,750/month

3. ACCURACY: "Needle in a haystack" problem
             → LLMs get WORSE at finding specific info buried in huge contexts
             → Research shows accuracy drops for facts in the middle of long contexts
```

**RAG's job:** Find the 3-5 most relevant chunks from your 12.5M token corpus and give ONLY those to the LLM.

---

### 2. THE RAG PIPELINE (End-to-End)

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE (End-to-End)                     │
│                                                                  │
│  OFFLINE (One-time setup):                                       │
│  ┌────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │Documents│ → │ CHUNKING │ → │ EMBEDDING │ → │ VECTOR STORE │  │
│  │(PDFs,   │   │(break    │   │(convert   │   │(store for    │  │
│  │ docs)   │   │ into     │   │ chunks to │   │ fast search) │  │
│  └────────┘   │ pieces)  │   │ vectors)  │   └──────────────┘  │
│               └──────────┘   └───────────┘                      │
│                                                                  │
│  ONLINE (Every query):                                           │
│  ┌───────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐   │
│  │ Query │ → │ RETRIEVAL │ → │RERANKING │ → │ GENERATION   │   │
│  │       │   │(find top  │   │(refine   │   │(LLM answers  │   │
│  └───────┘   │ matches)  │   │ ranking) │   │ with context) │   │
│               └───────────┘   └──────────┘   └──────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

Session 4.1 focuses on **CHUNKING** — the first step. If you chunk badly, everything downstream fails. Garbage chunks → bad embeddings → irrelevant retrieval → hallucinated answers.

---

### 3. THE NAIVE ALTERNATIVE — Why Not Just Use Fixed-Size Chunks?

The simplest approach: split every N characters.

```python
CHUNK_SIZE = 300
OVERLAP = 50

start = 0
while start < len(document):
    end = start + CHUNK_SIZE
    chunk = document[start:end]          # just slice!
    chunks.append(chunk)
    start = end - OVERLAP                # slide forward with overlap
```

**What goes wrong:**

```
Original text: "...Acme Corp reported strong Q3 results, with revenue reaching $4.2 billion,
                a 15% increase year-over-year. Net income was $890 million..."

Fixed-size cut at 300 chars:
  Chunk 3: "...with revenue reaching $4.2 billi"    ← CUT MID-WORD!
  Chunk 4: "illion, a 15% increase year-over-year..." ← context lost!

Table example:
  Chunk 5: "| Revenue | $4.2B | $3.65B |"        ← header row in previous chunk!
  The table data is meaningless without the header row.
```

**The overlap** (50 chars shared between adjacent chunks) is a band-aid — it helps sometimes but doesn't fix the fundamental problem of semantically incoherent chunks.

> 💡 **Interview tip:** "Fixed-size chunking is fine for prototyping but unacceptable in production because it breaks semantic boundaries — sentences, paragraphs, tables, and sections get split arbitrarily."

---

### 4. THE FIVE CHUNKING STRATEGIES

#### Strategy 1: Fixed-Size Chunking

**How:** Slide a window of N characters across the text, with optional overlap.

**Implementation:**
```python
CHUNK_SIZE = 300
OVERLAP = 50

chunks = []
start = 0
while start < len(text):
    end = start + CHUNK_SIZE
    chunk = text[start:end]
    chunks.append(chunk)
    start = end - OVERLAP
```

**Pros:** Dead simple, predictable chunk sizes, fast.
**Cons:** Breaks mid-sentence, mid-paragraph, mid-table.
**Use when:** Quick prototype, unstructured text with no headers.

---

#### Strategy 2: Recursive Character Splitting

**How:** Try to split on the best separator first, falling back to worse ones if chunks are still too big.

**Separator priority:**
```
\n\n  →  \n  →  ". "  →  " "
 ↑       ↑      ↑       ↑
paras   lines   sent.   words    (tries best separators first)
```

**Implementation (this is what LangChain's RecursiveCharacterTextSplitter does):**
```python
def recursive_split(text, chunk_size=400, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            current_chunk = ""
            chunks = []
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate          # fits → keep accumulating
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())  # save full chunk
                    if len(part) > chunk_size and i + 1 < len(separators):
                        # Part is still too big → recurse with next separator
                        sub_chunks = recursive_split(part, chunk_size, separators[i+1:])
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part           # start new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

    # Fallback: hard split at chunk_size
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]
```

**How the recursion works — traced on our sample:**
1. First tries `\n\n` — splits into paragraphs
2. Accumulates paragraphs until `chunk_size` (400 chars) is reached
3. If a single paragraph exceeds 400 chars → **recurse** with `\n` as separator
4. If a single line exceeds 400 chars → recurse with `. ` (sentences)
5. Last resort: split on spaces

**Pros:** Respects paragraph boundaries, widely used (LangChain default).
**Cons:** Doesn't understand document structure — doesn't know `## Revenue` is a section header.
**Use when:** General unstructured text (blog posts, articles, chat logs).

---

#### Strategy 3: Semantic Chunking

**How:** Embed each sentence into a vector, measure cosine similarity between adjacent sentences, split where similarity drops sharply (= topic boundary).

**Step-by-step:**
```
Step 1: Split text into sentences
        S1: "Cloud revenue grew 28% YoY."
        S2: "This was driven by enterprise AI adoption."
        S3: "Hardware revenue declined 8%."

Step 2: Embed each sentence
        S1 → [0.82, 0.15, 0.63, ...]
        S2 → [0.79, 0.18, 0.60, ...]
        S3 → [0.21, 0.88, 0.34, ...]

Step 3: Compute cosine similarity between ADJACENT sentences
        sim(S1, S2) = 0.94   ← HIGH! same topic (cloud)
        sim(S2, S3) = 0.31   ← LOW!  topic changed    ← BREAKPOINT

Step 4: Split where similarity drops below threshold
        Chunk 1: [S1, S2]    ← cloud topic
        Chunk 2: [S3, ...]   ← hardware topic
```

**Visual:**
```
Similarity between adjacent sentences:

  S1-S2    S2-S3    S3-S4    S4-S5    S5-S6    S6-S7
  0.94     0.31     0.88     0.91     0.29     0.85
  ████     █        ████     ████     █        ████
  ████     █        ████     ████     █        ████
  ████     ▼        ████     ████     ▼        ████
           BREAK                      BREAK

Result: [S1,S2] | [S3,S4,S5] | [S6,S7]
        chunk 1    chunk 2      chunk 3
```

**Pros:** Actually understands topic boundaries based on meaning, not formatting.
**Cons:** Requires calling an embedding model for every sentence at ingestion time (cost), needs a good threshold.
**Use when:** Unstructured text with no headers — transcripts, raw notes, chat logs.
**Not needed when:** Documents already have clear section headers (use structure-aware instead — cheaper and equally effective).

> 💡 **Interview tip:** "Semantic chunking detects topic boundaries using embedding similarity. It's ideal for unstructured text, but for documents with clear headers like financial reports, structure-aware chunking is cheaper and equally effective — you don't need an embedding model to know that 'Revenue Breakdown' and 'Risk Factors' are different topics."

---

#### Strategy 4: Document-Structure Aware Chunking

**How:** Parse the document's structure (markdown headers, HTML tags, PDF sections) and chunk by section. Each chunk carries a **breadcrumb** showing its location in the hierarchy.

**Implementation:**
```python
def structure_aware_split(text):
    chunks = []
    current_headers = {1: "", 2: "", 3: ""}    # tracks header hierarchy
    current_content = ""

    for line in text.split('\n'):
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if header_match:
            # Save accumulated content as a chunk
            if current_content.strip():
                breadcrumb = " > ".join(
                    current_headers[l] for l in [1,2,3] if current_headers[l]
                )
                chunks.append({"header": breadcrumb, "content": current_content.strip()})

            # Update header hierarchy
            level = len(header_match.group(1))     # # = 1, ## = 2, ### = 3
            current_headers[level] = header_match.group(2)
            for l in range(level + 1, 4):           # clear lower-level headers
                current_headers[l] = ""
            current_content = ""
        else:
            current_content += line + "\n"

    # Save final chunk
    if current_content.strip():
        breadcrumb = " > ".join(
            current_headers[l] for l in [1,2,3] if current_headers[l]
        )
        chunks.append({"header": breadcrumb, "content": current_content.strip()})
    return chunks
```

**Output on our sample financial report:**
```
Chunk 1: [Acme Corp > Executive Summary]
         "Acme Corp reported strong Q3 results, with revenue reaching $4.2 billion..."
         253 chars

Chunk 2: [Acme Corp > Revenue Breakdown > Cloud Services]
         "Cloud revenue grew 28% YoY to $2.1 billion..."
         202 chars

Chunk 3: [Acme Corp > Revenue Breakdown > Hardware Division]
         "Hardware revenue declined 8% to $1.4 billion..."
         226 chars

Chunk 4: [Acme Corp > Revenue Breakdown > Professional Services]
         "Professional services revenue was $700 million..."
         142 chars

Chunk 5: [Acme Corp > Key Metrics]
         "| Metric | Q3 2025 | Q3 2024 | Change |..."   ← table stays intact!
         267 chars

Chunk 6: [Acme Corp > Risk Factors]
         "The company faces several risks..."
         401 chars

Chunk 7: [Acme Corp > Forward Guidance]
         "For Q4 2025, the company expects..."
         251 chars
```

**Key design decisions in the code:**
- **Header hierarchy clearing (line: `for l in range(level + 1, 4): current_headers[l] = ""`):** When you hit `## Risk Factors`, any `###` headers from the previous `## Revenue Breakdown` section get cleared. Without this, you'd get a stale breadcrumb like `Report > Revenue > Cloud Services > Risk Factors`.
- **Breadcrumb as context:** The header `"Acme Corp > Revenue Breakdown > Cloud Services"` gives the LLM crucial context even without seeing the rest of the document.
- **Tables stay intact:** Because we split on headers, not characters, the metrics table stays in one chunk.

**Pros:** Each chunk is a coherent section with navigable header context.
**Cons:** Requires parsing (markdown, HTML, PDF). Section sizes vary wildly — an executive summary might be 100 chars while risk factors are 2000 chars.
**Use when:** Structured documents (financial reports, technical docs, legal contracts, wikis).

---

#### Strategy 5: Parent-Child Chunking

**How:** Create SMALL chunks for vector search (precise matching), but map each back to its LARGER parent section (rich context for the LLM). This resolves the fundamental chunk-size trade-off.

**The insight:**
```
SMALL chunks → embeddings are more specific → better retrieval precision
LARGE chunks → more context for LLM → better answer generation

Parent-child gives you BOTH:
  Search on small children, send large parents to the LLM.
```

**Implementation (builds on structure-aware):**
```python
def parent_child_split(text, child_size=150):
    parents = structure_aware_split(text)    # reuses Strategy 4!

    all_children = []
    for parent_idx, parent in enumerate(parents):
        # Split each parent into small sentence-level children
        sentences = re.split(r'(?<=[.!?])\s+', parent['content'])
        child_text = ""
        for sentence in sentences:
            if len(child_text) + len(sentence) > child_size and child_text:
                all_children.append({
                    "child_text": child_text.strip(),      # small → for search
                    "parent_idx": parent_idx,               # link to parent
                    "parent_header": parent['header'],      # breadcrumb
                    "parent_content": parent['content']      # large → for LLM
                })
                child_text = sentence
            else:
                child_text += " " + sentence if child_text else sentence
        if child_text.strip():
            all_children.append({...})  # save last child
    return all_children
```

**The regex `(?<=[.!?])\s+`:** A lookbehind that splits AFTER sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace. Splits on sentence boundaries without eating the punctuation.

**Output on our sample:**
```
Child 1:  "Acme Corp reported strong Q3 results, with revenue reaching $4.2 billion..."
          ↳ Parent: [Executive Summary] (253 chars)

Child 3:  "Cloud revenue grew 28% YoY to $2.1 billion."
          ↳ Parent: [Revenue > Cloud Services] (202 chars)

Child 4:  "Average contract value increased to $620K from $480K in Q3 2024."
          ↳ Parent: [Revenue > Cloud Services] (202 chars)
```

**Query-time flow:**
```
User: "What happened to cloud revenue?"

Step 1: SEARCH — Embed query, find closest child chunk
        → Match: "Cloud revenue grew 28% YoY to $2.1 billion."

Step 2: EXPAND — Look up parent via parent_idx
        → Parent: FULL Cloud Services section (202 chars)

Step 3: GENERATE — Send parent content + header breadcrumb to LLM
        → LLM has ALL the cloud services context to answer comprehensively
```

**Pros:** Best of both worlds — precise matching + rich context. Production-grade.
**Cons:** More complex implementation, need to store and track parent-child mappings in vector DB metadata.
**Use when:** Production systems where retrieval quality matters most.

---

### 5. COMPARISON TABLE

| Strategy | Splits Based On | Understands Topics? | Chunk Coherence | Best For | Complexity |
|----------|----------------|--------------------|--------------------|----------|------------|
| **Fixed-Size** | Character count | ❌ No | ❌ Breaks mid-sentence | Quick prototyping | Very Low |
| **Recursive Split** | `\n\n` → `\n` → `. ` → ` ` | ❌ No (format, not meaning) | 🟡 Mostly | General unstructured text | Low |
| **Semantic** | Embedding cosine similarity | ✅ Yes, always | ✅ Yes | Unstructured text with no headers | Medium |
| **Structure-Aware** | Headers / sections | ✅ Yes (if doc has headers) | ✅ Yes | Structured documents | Medium |
| **Parent-Child** | Headers + sentence split | ✅ Yes | ✅ Yes | Production systems | High |

---

### 6. THE KEY TRADE-OFF (Interview Favorite)

```
← Smaller chunks                              Larger chunks →

 Better retrieval precision                  More context for generation
 (embedding matches more specific queries)   (LLM has more info to work with)

 But: LLM gets less context                  But: more noise in retrieval
      may miss surrounding info                   embedding is less specific
```

**Parent-child resolves this:** small children for precision, large parents for context.

**The Golden Rule:**
> "A chunk should contain exactly ONE coherent idea, with enough context for the LLM to understand it without needing other chunks."

---

### 7. INTERVIEW ANGLES

**Q: "How do you choose a chunking strategy?"**
> "It depends on the document type. For structured documents like financial reports or legal contracts, I'd use structure-aware or parent-child chunking because these documents have clear headers and sections that carry semantic meaning. For unstructured text like transcripts, I'd use semantic chunking to detect topic boundaries using embedding similarity. The key trade-off is chunk size — smaller chunks give more precise retrieval but less context for generation. Parent-child chunking resolves this by using small chunks for search and returning the full parent section to the LLM. I always benchmark multiple strategies on the actual data because there's no universal best."

**Q: "Your RAG system is returning irrelevant chunks. What do you check first?"**
> "First, I'd check chunking — if chunks break mid-sentence or mix multiple topics, the embeddings become noisy and retrieval degrades. I'd switch to structure-aware or parent-child chunking to ensure each chunk is semantically coherent. Then I'd check whether the chunk size is appropriate for my embedding model — most models perform best with chunks of 100-500 tokens."

**Q: "What is the difference between recursive splitting and semantic chunking?"**
> "Recursive splitting uses text formatting (paragraph breaks, newlines, periods) as splitting boundaries — it's format-aware but not meaning-aware. Semantic chunking embeds each sentence and splits where cosine similarity between adjacent sentences drops — it's meaning-aware. Semantic chunking is better for text without clear formatting but costs more because you need to call an embedding model during ingestion."

---

### 8. CODE REFERENCE

See [chunking_experiments.py](file:///Users/shreyabharadwaj/Shreya_storage/Sourav/Machine%20Learning/AI%20Agents%20Learn/module_04_advanced_rag/chunking_experiments.py) for a runnable demonstration of all 4 implemented strategies on the same financial report document.

**Script output summary (from running the script):**
```
┌─────────────────────┬────────┬────────────┬───────────┬──────────────┐
│ Strategy            │ Chunks │ Avg Size   │ Coherent? │ Best For     │
├─────────────────────┼────────┼────────────┼───────────┼──────────────┤
│ 1. Fixed-Size       │     8  │   300 chars │ ❌ No     │ Quick & dirty│
│ 2. Recursive Split  │     8  │ ~400 chars │ 🟡 Mostly │ General text │
│ 3. Structure-Aware  │     7  │ Variable   │ ✅ Yes    │ Structured   │
│ 4. Parent-Child     │    14  │ ~150 chars │ ✅ Yes    │ Production   │
└─────────────────────┴────────┴────────────┴───────────┴──────────────┘
```

> **Note:** Semantic chunking (Strategy 3 in our discussion) is not implemented in the script because it requires an embedding model. It will be demonstrated in Session 4.2 when we set up embeddings.

---

### 9. COMMON MISCONCEPTIONS

| Misconception | Reality |
|---------------|---------|
| "Bigger chunks are always better because the LLM gets more context" | Bigger chunks have noisier embeddings → worse retrieval. The LLM never even sees the chunk if retrieval fails. |
| "Overlap solves the problem of fixed-size chunking" | Overlap is a band-aid. It helps if a relevant sentence gets split, but the overlapping text gets embedded twice (wasted compute) and chunks are still incoherent. |
| "Semantic chunking is always the best strategy" | It's expensive (embedding every sentence) and unnecessary when documents have clear structure. Structure-aware is cheaper and equally effective for formatted docs. |
| "You should use the same chunk size for all documents" | Different document types need different strategies. A 200-token chunk is great for FAQ entries but terrible for legal paragraphs that span 500+ tokens. |
