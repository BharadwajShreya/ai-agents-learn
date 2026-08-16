# Module 1 — Sessions 1.1 & 1.2: Tokenization, Embeddings & Positional Encoding
> **Revision Notes** — covers everything discussed in learning sessions. Use this for quick review before interviews.

---

## Part 1: Tokenization

### Why Do We Need Tokenization?
LLMs cannot read raw text — they operate on numbers. Tokenization converts text → integer IDs that can be looked up in an embedding table.

### Why Sub-Word Tokenization (Not Word-Level or Character-Level)?

| Strategy | Example for "apple" | Problem |
|----------|---------------------|---------|
| **Word-Level** | `apple` → `1` token | Vocabulary explodes (millions of words, names, slang). OOV errors for unseen words. |
| **Character-Level** | `apple` → `5` tokens (`a,p,p,l,e`) | Sequences become very long. Model loses semantic meaning (no difference between `app` and `apl`). |
| **Sub-Word (BPE)** ✅ | `apple` → `1` token, `tokenization` → `2-3` tokens | Goldilocks balance — compact vocab, no OOV errors, meaningful sub-units. |

---

### BPE (Byte Pair Encoding)
The most widely used sub-word tokenization algorithm (used by GPT-2/3/4, Llama 3, Mistral, etc.)

#### How BPE Builds Its Vocabulary:
BPE starts with individual characters and iteratively merges the most frequent adjacent pair.

**Example with training text: `["low", "lower", "newest"]`**

```
Step 0 — Start with characters:
Vocab = {l, o, w, e, r, n, s, t}
Text:   l o w | l o w e r | n e w e s t

Step 1 — Most frequent pair: (e, r) → merge into "er"
Vocab = {l, o, w, e, r, n, s, t, er}
Text:   l o w | l o w [er] | n e w e s t

Step 2 — Most frequent pair: (e, s) → merge into "es"
Vocab = {l, o, w, e, r, n, s, t, er, es}
Text:   l o w | l o w [er] | n e w [es] t

Step 3 — Most frequent pair: (es, t) → merge into "est"
Vocab = {l, o, w, e, r, n, s, t, er, es, est}
Text:   l o w | l o w [er] | n e w [est]
```

This process repeats thousands of times until the vocabulary reaches the target size (e.g., 50,000).

---

### Key Trade-off: Vocabulary Size vs. Sequence Length

> **Bigger vocab = shorter sequences = larger embedding table**

| Vocab Size | Effect on Sequence Length | Effect on Embedding Table |
|------------|--------------------------|--------------------------|
| Small (26 chars) | Very long (every word → many tokens) | Small table |
| Medium (32,000) | Moderate | Medium table (GPT-2 level) |
| Large (100,000) | Short (most words → single token) | Large table |

**The APPLE example:**
- Vocab = 26 letters only → `APPLE` = **5 tokens** (A, P, P, L, E)
- Vocab = 50,000 words → `APPLE` = **1 token**

So bigger vocabulary → fewer tokens to represent the same sentence → shorter sequences.

---

### Handling Unknown Words — No OOV Errors!
Modern sub-word tokenizers **never** produce an `<UNK>` (Unknown) token. Even a made-up word like `ChatGPTinator` gets broken into known sub-pieces:
```
ChatGPTinator → "Chat" + "G" + "PT" + "in" + "ator"
```

---

### Real-World Demo: Common vs. Rare Words (tiktoken / GPT-4)

```text
Input: "apple"
Number of tokens: 1
Token ID: 23182 → String: 'apple'   ← Common word = 1 token

Input: "otorhinolaryngology" (a real medical word)
Number of tokens: 6
Token ID: 10088 → String: 'otor'
Token ID: 42657 → String: 'hin'
Token ID:   337 → String: 'ol'
Token ID:   661 → String: 'ary'
Token ID:   983 → String: 'ng'
Token ID:  2508 → String: 'ology'  ← Rare word = 6 tokens!
```

> **Interview insight:** A 19-letter medical word consumes 6x the context window vs. a common word. This is why specialized LLMs (medical, legal, code) often train **domain-specific tokenizers** that include jargon as single tokens.

---

### Other Tokenizers to Know
| Tokenizer | Used By | Note |
|-----------|---------|------|
| **BPE** | GPT-2/3/4, Llama, Mistral | Most common |
| **WordPiece** | BERT, DistilBERT | Similar to BPE but uses likelihood-based merges |
| **SentencePiece** | T5, Gemma, Llama (wrapper) | Language-agnostic, works on raw bytes |
| **Unigram** | Some multilingual models | Probabilistic, keeps most likely sub-words |

---

## Part 2: Token Embeddings

### The Problem: Integers Don't Have Meaning
After tokenization, `"king"` is integer `7078`. But the model can't reason about the *number* 7078 — it doesn't know `7078` (king) is related to `7436` (queen) or `6260` (throne).

### The Solution: Embedding Table (Lookup Matrix)
The **Embedding Table** is a learnable matrix of shape:
```
Embedding Table shape = (vocab_size, d_model)
e.g.:                   (50,000,     4,096)
```

Each row is a vector of `d_model` floating-point numbers representing the "meaning" of that token. To get the embedding for a token, simply **look up its row** in the table:

```
Token ID 7078 ("king")  → Row 7078 → [ 0.91,  0.04,  0.77, -0.21, ...] (4096 numbers)
Token ID 7436 ("queen") → Row 7436 → [ 0.88, -0.11,  0.74, -0.18, ...] (4096 numbers)
                                       ↑ Similar! The model learned these are related words.
```

### What is `d_model` (Embedding Dimension)?
It's the length of each row vector — the "number of personality traits" per token.

| Model | d_model |
|-------|---------|
| GPT-2 Small | 768 |
| GPT-3 (175B) | 12,288 |
| Llama 3 8B | 4,096 |
| Llama 3 70B | 8,192 |

### Key Properties:
- **Initialized randomly** — starts as noise, becomes meaningful via training
- **Trained end-to-end** — learned via backpropagation during pre-training
- **Semantically meaningful** — similar words converge to similar vectors (king ≈ queen, cat ≈ dog)
- **Total parameters in embedding table:** `vocab_size × d_model` (e.g. `50,000 × 4,096` = **205M parameters** just for embeddings!)

### Step-by-Step: Input Sentence → Embeddings

For input sentence `"the cat sat"` (seq_len=3, d_model=4 for this demo):

```
Step 1 — Tokenize:  "the" → 2,  "cat" → 5,  "sat" → 7
          input_ids = [2, 5, 7]

Step 2 — Lookup rows from Embedding Table:
          Token ID 2 → [ 0.20,  0.42, -0.96,  0.94]  ("the")
          Token ID 5 → [ 0.22, -0.72, -0.42, -0.27]  ("cat")
          Token ID 7 → [ 0.18, -0.91,  0.22, -0.66]  ("sat")
          Result shape: (3, 4) = (seq_len, d_model)
```

---

## Part 3: Positional Encoding

### The Problem: Transformers Are Order-Blind
Transformers process **all tokens in parallel** (unlike RNNs which go left-to-right). This means they have no built-in concept of order.

Without positional encoding, these two sentences produce **identical** embeddings — the model can't tell them apart:
```
"The dog bit the man"
"The man bit the dog"
```

### Why Not Just Use 0, 1, 2, 3...?
Two major problems:
1. **Scale mismatch** — Embedding values are between `[-1, 1]`. Adding `1000` for position 1000 completely drowns out the semantic meaning. Normalizing to `pos/10000` makes positions `1000` and `1001` so similar (`0.1000` vs `0.1001`) the model can barely distinguish them.
2. **Generalization** — Neural networks struggle with numbers they've never seen. If training max length was 512, the model has no idea what position `513` means.

### Solution: Positional Encoding (PE)
Generate a unique vector (same shape as token embedding: `d_model`) for every position. Then **add** them together:

```
Final Input to Transformer = Token Embedding + Positional Encoding
```

---

### Type 1: Sinusoidal Encoding (Original 2017 "Attention Is All You Need")

**Core idea:** Represent each position as a vector of sine and cosine waves at different frequencies.

#### The Formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))   ← even dimensions use sine
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))   ← odd dimensions use cosine
```

#### Worked Example (d_model=4):
```
              Dim 0          Dim 1          Dim 2          Dim 3
              (fast sine)    (fast cosine)  (slow sine)    (slow cosine)
Pos 0 (the):  sin(0) = 0.00  cos(0) = 1.00  sin(0) = 0.00  cos(0) = 1.00
Pos 1 (cat):  sin(1) = 0.84  cos(1) = 0.54  sin(≈0) = 0.01 cos(≈0) = 1.00
Pos 2 (sat):  sin(2) = 0.91  cos(2) =-0.42  sin(≈0) = 0.02 cos(≈0) = 1.00
```

#### The Clock Analogy:
- **Dim 0 (seconds hand):** Oscillates very fast — changes a lot from position 0 → 1
- **Dim 2 (minutes hand):** Oscillates slowly — barely changes from position 0 → 1, but differentiates large distances
- **Dim 4 (hours hand):** Oscillates very slowly — captures macro-level position

By combining fast + medium + slow waves, every position gets a *unique, bounded* fingerprint vector.

#### Pros & Cons:
| ✅ Pros | ❌ Cons |
|---------|---------|
| No learned parameters | Encodes **absolute** position (not relative) |
| Can theoretically extrapolate | Struggles with sequences longer than training max |
| Bounded between -1 and 1 | Model must learn relative distance indirectly |

---

### Type 2: RoPE — Rotary Position Embedding (2024+ Standard)
> Used by: **Llama 2, Llama 3, Mistral, Gemma, Qwen, GPT-NeoX, Falcon**

#### The Key Insight: Language Cares About Relative Distance
`"red"` modifies the noun right next to it — whether they're at positions (5, 6) or (1005, 1006). The *absolute* position doesn't matter; only the *gap* does. Sinusoidal encoding doesn't explicitly model this.

#### How RoPE Works: Rotation Instead of Addition
Split the `d_model` dimension into pairs `(x, y)`. Each pair is like a 2D point on a circle. Instead of *adding* a position vector, RoPE *rotates* the pair by an angle proportional to the position:

```
Pos 0 (no rotation):          Pos 1 (rotate by θ):           Pos 2 (rotate by 2θ):
        (0,1)                          (y)                            (y)
          ▲                             ▲                              ▲
          │                             │  ● (x,y)                    │
          │                             │ /                            │     ● (x,y)
          │                             │/ θ                           │   / 2θ
          └────► (1,0)                  └────► (x)                     └────► (x)
```

- **Position 0** → no rotation (angle 0)
- **Position 1** → rotate by θ (e.g. 10°)
- **Position 100** → rotate by 100θ (e.g. 1000°)

#### The Math Magic (Why Relative Distance Falls Out):
In transformers, tokens interact via the **dot product** of their Query and Key vectors.

When you compute `Query(pos=m) · Key(pos=n)` where both are rotated by RoPE:

```
Dot Product = f(m, n) = cos((m - n)θ)
```

The absolute positions `m` and `n` cancel out! The interaction score depends **only on `(m - n)`** — the relative gap between them.

**Practical consequence:**
- Token at position 5 and token at position 6 interact with the exact same strength as tokens at position 1005 and 1006
- This is why Llama 3 can handle 128K context windows even when trained on shorter sequences

#### Pros & Cons:
| ✅ Pros | ❌ Cons |
|---------|---------|
| Explicitly models **relative** distance | Slightly more complex to implement |
| Excellent extrapolation to long contexts | — |
| No extra learned parameters | — |
| Now the industry standard | — |

---

### How Token Embeddings + Positional Encoding Come Together
The full addition, step-by-step, with real numbers from our demo (seq_len=3, d_model=4):

```
Token Embeddings (3, 4):              Positional Encodings (3, 4):     Final Embeddings (3, 4):
"the"  [ 0.20,  0.42, -0.96,  0.94]  +  [0.00,  1.00,  0.00,  1.00]  =  [ 0.20,  1.42, -0.96,  1.94]
"cat"  [ 0.22, -0.72, -0.42, -0.27]  +  [0.84,  0.54,  0.01,  1.00]  =  [ 1.07, -0.18, -0.41,  0.73]
"sat"  [ 0.18, -0.91,  0.22, -0.66]  +  [0.91, -0.42,  0.02,  1.00]  =  [ 1.09, -1.32,  0.24,  0.34]
```

Each final vector carries **both**:
- **Semantic meaning** (what the word means) from the token embedding
- **Position information** (where the word sits) from the positional encoding

These final vectors (shape `seq_len × d_model`) are what flows into the first transformer layer.

---

## Quick Interview Reference Card

| Concept | Key Answer |
|---------|-----------|
| Why sub-word tokenization? | Balance between word-level (OOV, huge vocab) and character-level (too many tokens, no semantics) |
| Bigger vocab → sequence length? | **Decreases** (more words stored as single tokens) |
| OOV errors in BPE? | **Never** — falls back to smaller sub-words |
| Domain-specific tokenizer (e.g. medical)? | Rare domain words get compressed to single tokens → shorter sequences, better understanding |
| What is d_model? | The dimension of each token's vector (e.g. 4096 for Llama 3 8B) |
| Embedding table size? | `vocab_size × d_model` |
| Why positional encoding? | Transformers process all tokens in parallel — no built-in order |
| Why not integers (0,1,2...)? | Scale mismatch + generalization failure |
| Sinusoidal pros? | No learned params, bounded values, deterministic |
| Sinusoidal cons? | Absolute position only, struggles on sequences longer than training max |
| RoPE advantage? | Explicitly models **relative** distance; excellent extrapolation to long contexts |
| Who uses RoPE? | Llama 2/3, Mistral, Gemma, Qwen, GPT-NeoX |

---

## Code References
- `mini_gpt/tokenizer.py` — Live demo comparing GPT-4 tokenization of common vs. medical words
- `mini_gpt/embedding.py` — Step-by-step demo of embedding lookup + sinusoidal PE addition with real numbers

### embedding.py Output (seq_len=3 "the cat sat", d_model=4, vocab_size=10)
```text
=== 1. EMBEDDING TABLE (vocab_size x d_model) ===
Shape: (10, 4)
Token ID 2 vector: [ 0.2   0.42 -0.96  0.94]   ← "the"
Token ID 5 vector: [ 0.22 -0.72 -0.42 -0.27]   ← "cat"
Token ID 7 vector: [ 0.18 -0.91  0.22 -0.66]   ← "sat"

=== 2. TOKEN EMBEDDINGS (seq_len x d_model) ===
Shape: (3, 4)        ← only the 3 rows we needed from the table

=== 3. POSITIONAL ENCODINGS (seq_len x d_model) ===
Shape: (3, 4)
Pos 0 Encoder: [0.   1.   0.   1.  ]   ← sin(0)=0, cos(0)=1 for all dims
Pos 1 Encoder: [0.84 0.54 0.01 1.  ]   ← Dim 0 changed a LOT (fast wave). Dim 2 barely changed (slow wave).
Pos 2 Encoder: [0.91 -0.42 0.02 1. ]   ← Same pattern, fast dim still moving fast.

=== 4. FINAL EMBEDDINGS (Token + Position) ===
Shape: (3, 4)
Final Pos 0: [ 0.20,  1.42, -0.96,  1.94]  ← "the" at position 0
Final Pos 1: [ 1.07, -0.18, -0.41,  0.73]  ← "cat" at position 1
Final Pos 2: [ 1.09, -1.32,  0.24,  0.34]  ← "sat" at position 2
```
**Key observation:** Even if we reused the same word at two different positions, its final vector would be different because the positional encoding added is unique per position.

---
---

# Session 1.3: Self-Attention Mechanism
> **Revision Notes** — the core mechanism that makes Transformers work. Master this for interviews.

---

## Part 1: What Is Self-Attention?

### The Intuition: A Room Full of People
Imagine someone says: *"The animal didn't cross the street because **it** was too tired."*

When you hear **"it"**, your brain asks: *"What does 'it' refer to?"* You look at all the other words and decide **"animal"** is most relevant. You pay high attention to "animal" and low attention to "street."

**Self-attention does exactly this.** Every token looks at **every other token** in the sequence (including itself) and computes: *"How relevant is each of you to me?"*

### Why Self-Attention Matters
- **Before Transformers (RNNs):** Information had to travel sequentially through the network. Token 1 → Token 2 → … → Token 100. By the time you reach Token 100, information about Token 1 is diluted.
- **With Self-Attention:** Every token can directly attend to every other token in **one step**, regardless of distance. Token 100 can look directly at Token 1.

---

## Part 2: The Three Players — Query, Key, Value

### The Library Analogy
| Role | Analogy | What It Is |
|------|---------|-----------|
| **Query (Q)** | You walk into a library: *"I need info about X"* | The question each token asks |
| **Key (K)** | Labels on every book's spine | What each token advertises about itself |
| **Value (V)** | The actual content inside each book | The information each token provides |

The process:
1. Compare your **Query** against every **Key** → relevance scores
2. Normalize scores with softmax → attention weights (probabilities)
3. Take a weighted mix of **Values** → context-enriched output

---

## Part 3: The Math, Step by Step

### Creating Q, K, V
Given input embeddings `X` of shape `(seq_len, d_model)`:

```
Q = X · W_Q     shape: (seq_len, d_k)      ← "What am I looking for?"
K = X · W_K     shape: (seq_len, d_k)      ← "What do I contain?"
V = X · W_V     shape: (seq_len, d_v)      ← "What info can I provide?"
```

`W_Q`, `W_K`, `W_V` are **learned** weight matrices of shape `(d_model, d_k)`.

### The Complete Formula
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

### Step-by-Step Breakdown

```
Step 1:  Q = X · W_Q,  K = X · W_K,  V = X · W_V
         ↓
Step 2:  Scores = Q · K^T                          shape: (seq_len, seq_len)
         Scores[i][j] = "how relevant is token j to token i?"
         ↓
Step 3:  Scaled Scores = Scores / √d_k             ← prevents softmax saturation
         ↓
Step 4:  Attention Weights = softmax(Scaled Scores) ← each row sums to 1
         ↓
Step 5:  Output = Attention Weights · V             shape: (seq_len, d_v)
         ↓
         Each token's output is a "context-enriched" vector
```

---

## Part 4: Worked Example — "the cat sat"

Using final embeddings from Session 1.2 as input `X` (seq_len=3, d_model=4):

### Step 1 — Create Q, K, V
Multiply input X by learned weight matrices W_Q, W_K, W_V (each of shape `d_model × d_k`):
```
Q = X · W_Q  →  shape (3, 4)
  Q["the"] = [ 0.192,  0.641, -1.080, -0.499]
  Q["cat"] = [-0.146,  0.198,  0.208, -0.544]
  Q["sat"] = [ 0.117, -0.455,  0.959, -0.374]

K = X · W_K  →  shape (3, 4)
  K["the"] = [ 0.726,  0.448, -0.748, -0.693]
  K["cat"] = [ 0.792,  0.511,  0.205, -0.377]
  K["sat"] = [ 0.325, -0.009,  0.738, -0.728]

V = X · W_V  →  shape (3, 4)
  V["the"] = [ 0.324,  0.742,  1.186,  0.185]
  V["cat"] = [-0.396, -0.052,  0.917, -0.051]
  V["sat"] = [-0.604, -0.627,  0.867, -0.086]
```

Each token now has a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what info can I provide?").

### Step 2 — Raw Attention Scores (Q · K^T)
```
               "the"   "cat"   "sat"
    "the"   [  1.58,   0.45,  -0.38 ]   ← "the" finds itself most relevant
    "cat"   [  0.20,   0.23,   0.50 ]   ← "cat" finds "sat" slightly more relevant
    "sat"   [ -0.58,   0.20,   1.02 ]   ← "sat" strongly attends to itself
```

### Step 3 — Scale by √d_k = √4 = 2.0
```
               "the"   "cat"   "sat"
    "the"   [  0.79,   0.22,  -0.19 ]
    "cat"   [  0.10,   0.12,   0.25 ]
    "sat"   [ -0.29,   0.10,   0.51 ]
```

### Step 4 — Softmax → Attention Weights
```
               "the"   "cat"   "sat"
    "the"   [  0.52,   0.29,   0.19 ]   sum = 1.00  ← probability distribution!
    "cat"   [  0.32,   0.32,   0.37 ]   sum = 1.00
    "sat"   [  0.21,   0.31,   0.47 ]   sum = 1.00
```

**Reading this map:**
- "the" puts 52% of its attention on itself, 29% on "cat", 19% on "sat"
- "cat" distributes attention roughly evenly across all three tokens
- "sat" focuses most on itself (47%) but also attends to "cat" (31%)

### Step 5 — Weighted Sum of Values
```
Output["the"] = 0.52 × V_the + 0.29 × V_cat + 0.19 × V_sat
Output["cat"] = 0.32 × V_the + 0.32 × V_cat + 0.37 × V_sat
Output["sat"] = 0.21 × V_the + 0.31 × V_cat + 0.47 × V_sat
```

Each output vector is now **context-enriched** — it carries information from all the tokens it attended to, not just its own embedding.

---

## Part 5: Why Scale by √d_k? (Critical Interview Question)

### The Problem: Softmax Saturation
When `d_k` is large (e.g., 128), the dot product `Q · K^T` is the sum of 128 multiplications — producing very large numbers. Large inputs push softmax into its **saturated region** where it outputs ~1 for the max and ~0 for everything else:

```
Without scaling (d_k = 128):
  Scores:  [45.2, 3.1, -12.7]
  softmax: [1.00, 0.00,  0.00]    ← ALL weight on one token! Gradient ≈ 0.

With scaling (÷ √128 ≈ 11.3):
  Scores:  [ 4.0, 0.27,  -1.1]
  softmax: [0.79, 0.19,  0.05]    ← Smooth distribution, healthy gradients ✅
```

### Why √d_k Specifically?
If Q and K entries are independent with mean 0 and variance 1, then their dot product has:
- **Mean = 0**
- **Variance = d_k** (sum of d_k independent products)

Dividing by `√d_k` brings the variance back to 1, keeping the scores in a range where softmax behaves well.

> **One-sentence interview answer:** *"Scaling by √d_k normalizes the variance of the dot products back to 1 regardless of dimension, preventing softmax saturation and maintaining healthy gradient flow."*

---

## Part 6: Causal (Masked) Self-Attention

### Why Causal Masking?
In GPT-style (decoder-only) models, tokens must **only attend to past tokens** (and themselves). During training, we process entire sequences in parallel, but the model shouldn't "cheat" by looking at future tokens.

### How It Works
Apply a **causal mask** before softmax — set future positions to `-∞` so they become 0 after softmax:

```
Causal Mask (✓ = allowed, ✗ = blocked):
    "the":  ✓  ✗  ✗     ← can only see itself
    "cat":  ✓  ✓  ✗     ← can see "the" and itself
    "sat":  ✓  ✓  ✓     ← can see everything up to itself
```

```
Scores AFTER masking:                    Attention weights AFTER softmax:
         "the"  "cat"   "sat"                    "the"  "cat"   "sat"
"the"  [  0.79,  -∞,     -∞  ]           "the"  [ 1.00, 0.00,  0.00 ]
"cat"  [  0.10,  0.12,   -∞  ]    →      "cat"  [ 0.50, 0.50,  0.00 ]
"sat"  [ -0.29,  0.10,   0.51]           "sat"  [ 0.21, 0.31,  0.47 ]
```

**Key insight:** "the" can only see itself (100% self-attention). As we move later in the sequence, tokens gain access to more context — exactly matching how autoregressive generation works at inference time.

### When to Use Which?
| Attention Type | Mask | Used In | Task |
|---------------|------|---------|------|
| **Bidirectional** (no mask) | None | BERT (encoder-only) | Understanding, classification |
| **Causal** (lower triangular) | Future tokens masked | GPT (decoder-only) | Text generation |
| **Cross-Attention** | Varies | T5, BART (encoder-decoder) | Translation, summarization |

---

## Part 7: Computational Complexity

### The O(n²) Problem
The attention score matrix `Q · K^T` has shape `(seq_len, seq_len)`. For a sequence of `n` tokens:
- **Memory:** O(n²) — storing the full attention matrix
- **Compute:** O(n² · d_k) — computing all pairwise dot products

| seq_len | Attention Matrix Size | Memory (FP16) |
|---------|----------------------|---------------|
| 512 | 262K cells | ~0.5 MB |
| 2,048 | 4.2M cells | ~8 MB |
| 8,192 | 67M cells | ~128 MB |
| 128,000 | 16.4B cells | ~31 GB |

> **Interview insight:** This quadratic cost is why long-context models are expensive, and why efficient attention variants (FlashAttention, sparse attention, linear attention) are active research areas.

---

## Quick Interview Reference Card — Self-Attention

| Concept | Key Answer |
|---------|-----------| 
| What are Q, K, V? | Q = "what am I looking for?", K = "what do I contain?", V = "what info can I provide?" — all created by multiplying input X with learned weight matrices |
| Shape of attention matrix? | `(seq_len, seq_len)` — each row is a probability distribution |
| Why scale by √d_k? | Prevents dot products from growing with d_k, which would saturate softmax and kill gradients |
| What is causal masking? | Lower-triangular mask that blocks attention to future tokens — enforces autoregressive property |
| BERT vs GPT attention? | BERT = bidirectional (no mask). GPT = causal (masked, can only see past) |
| Complexity of self-attention? | O(n²) in both time and memory — this is the bottleneck for long sequences |
| Why is attention interpretable? | The attention weight matrix directly shows which tokens the model is "looking at" — can be visualized as a heatmap |
| What does self-attention output? | Context-enriched vectors — each token's representation is now a weighted mix of all relevant tokens' values |

---

## Code References
- `mini_gpt/attention.py` — Full self-attention implementation with worked example, causal masking, and detailed step-by-step output

### attention.py Key Output
```text
Attention Weights (no mask):
         "the"    "cat"    "sat"
"the"    0.515    0.292    0.193   (sum=1.00)
"cat"    0.315    0.320    0.365   (sum=1.00)
"sat"    0.213    0.314    0.474   (sum=1.00)

Attention Weights (causal mask):
         "the"    "cat"    "sat"
"the"    1.000    0.000    0.000   (sum=1.00)  ← only sees itself
"cat"    0.496    0.504    0.000   (sum=1.00)  ← sees "the" + itself
"sat"    0.213    0.314    0.474   (sum=1.00)  ← sees all past tokens
```

---
---

# Session 1.4: Multi-Head Attention
> **Revision Notes** — why one attention head isn't enough, and how multiple heads capture different relationships in parallel.

---

## Part 1: Why Multiple Heads?

### The Problem with Single-Head Attention
A single attention head computes **one set** of Q, K, V — it can only learn **one pattern** of attention. But language has many simultaneous relationships:

| Relationship Type | Example: "The cat, which was very hungry, ate the fish." |
|-------------------|--------------------------------------------------------|
| **Syntactic** | "cat" → "ate" (subject-verb agreement) |
| **Coreference** | "which" → "cat" (pronoun resolution) |
| **Modifier** | "very" → "hungry" (adverb → adjective) |
| **Semantic role** | "ate" → "fish" (verb → object) |

**Multi-head attention = multiple pairs of eyes**, each specializing in a different relationship type, all in parallel.

---

## Part 2: How It Works — Split → Attend → Concatenate → Project

### The Key Trick: Same Compute, More Perspectives
Instead of one big attention head with `d_k = d_model`:
```
Single Head:   d_k = 512            → 1 attention pattern
Multi-Head:    d_k = 512/8 = 64     → 8 attention patterns!  (same total compute)
```

### Architecture Flow
```
Input X:  (seq_len, d_model)     e.g., (100, 512)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
  Head 1          Head 2    ...   Head h
  Q₁K₁V₁         Q₂K₂V₂         QₕKₕVₕ
  (100,64)        (100,64)        (100,64)
    │               │               │
  Attention       Attention       Attention
  (100,64)        (100,64)        (100,64)
    │               │               │
    └───────────────┼───────────────┘
                    ▼
              Concatenate
            (100, 512)     ← h × d_k = d_model
                    │
              W_O (linear)
            (100, 512)     ← Final output
```

### The Math
```
For each head i (where i = 1 to h):
    Q_i = X · W_Q_i     W_Q_i shape: (d_model, d_k)
    K_i = X · W_K_i     W_K_i shape: (d_model, d_k)
    V_i = X · W_V_i     W_V_i shape: (d_model, d_v)
    head_i = Attention(Q_i, K_i, V_i)

MultiHead(X) = Concat(head_1, ..., head_h) · W_O
               (seq_len, h × d_v)  ·  (h × d_v, d_model)  =  (seq_len, d_model)
```

---

## Part 3: Real-World Dimensions

| Model | d_model | Heads (h) | d_k = d_model/h |
|-------|---------|-----------|------------------|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-3 (175B) | 12,288 | 96 | 128 |
| Llama 3 8B | 4,096 | 32 | 128 |
| Llama 3 70B | 8,192 | 64 | 128 |

---

## Part 4: What Different Heads Learn

Research shows that different heads **specialize**:

| Head Pattern | What It Learns | Attention Visualization |
|-------------|----------------|------------------------|
| Positional | Attend to previous/next token | Strong diagonal pattern |
| Syntactic | Subject → verb agreement | Long-range links across clauses |
| Coreference | Pronoun → antecedent | "it" → "cat", "they" → "students" |
| Structural | Attend to punctuation/separators | Focus on commas, periods |

### Demo Output — Head 1 vs Head 2 on "the cat sat"
```
Head 1 Attention Weights:            Head 2 Attention Weights:
         "the"  "cat"  "sat"                  "the"  "cat"  "sat"
"the"    0.063  0.247  0.690          "the"    0.495  0.294  0.211
"cat"    0.266  0.324  0.410          "cat"    0.451  0.305  0.245
"sat"    0.479  0.290  0.230          "sat"    0.377  0.324  0.299
```

Head 1: "the" focuses on "sat" (69%) — forward-looking pattern
Head 2: "the" focuses on itself (49.5%) — positional/self-reference pattern

**Different heads, different perspectives!**

---

## Part 5: The Output Projection (W_O) — Often Overlooked

After concatenating all heads:
```
Concat: (seq_len, h × d_v)  →  W_O: (h × d_v, d_model)  →  Output: (seq_len, d_model)
```

**Why is W_O needed?**
- Each head operates in its own isolated subspace
- Concatenation just stacks them side by side
- `W_O` **mixes information across heads** — combining insights from different "perspectives" into a unified representation
- Without `W_O`, the model couldn't learn cross-head interactions

---

## Part 6: Parameter Count — Single vs Multi-Head

The total parameter count is approximately the same:

```
Single-Head (d_k = d_model = 512):
  W_Q + W_K + W_V = 3 × (512 × 512) = 786,432 params
  1 attention pattern

Multi-Head (8 heads, d_k = 64):
  8 × (W_Q_i + W_K_i + W_V_i) = 8 × 3 × (512 × 64) = 786,432 params
  + W_O = 512 × 512 = 262,144 params
  Total: 1,048,576 params, 8 attention patterns
```

> **Interview insight:** Multi-head costs slightly more (due to W_O), but the quality improvement is massive. It's the difference between asking 1 generalist vs. 8 specialists and combining their answers.

---

## Quick Interview Reference Card — Multi-Head Attention

| Concept | Key Answer |
|---------|-----------|
| Why multi-head? | Each head learns different relationship types (syntax, coreference, position) in parallel |
| How does it work? | Split d_model into h heads of size d_k, run independent attention, concatenate, project with W_O |
| d_k per head? | d_k = d_model / h (e.g., 4096/32 = 128 for Llama 3) |
| Parameter count vs single? | ~Same for Q/K/V, slightly more with W_O — but h× richer representation |
| What is W_O? | Output projection that mixes information across heads after concatenation |
| Why not one big head? | Single head = one attention pattern. Multi-head = h patterns. Empirically far superior |
| Real example? | GPT-2: 12 heads, GPT-3: 96 heads, Llama 3: 32-64 heads |

---

## Code References
- `mini_gpt/multi_head_attention.py` — Full multi-head attention with split/attend/concat/project, head comparison, and parameter count analysis

### multi_head_attention.py Key Output (seq_len=3 "the cat sat", d_model=4, h=2 heads, d_k=2)
```text
⚙️  Config: h=2 heads, d_k = d_model/h = 4/2 = 2
   Each head works in a 2-dimensional subspace

Head 1 Attention Weights:            Head 2 Attention Weights:
         "the"  "cat"  "sat"                  "the"  "cat"  "sat"
"the"    0.063  0.247  0.690          "the"    0.495  0.294  0.211
"cat"    0.266  0.324  0.410          "cat"    0.451  0.305  0.245
"sat"    0.479  0.290  0.230          "sat"    0.377  0.324  0.299

📊 Comparison — who does each token attend to most?
  "the" in Head 1: focuses on "sat" (69.0%)    ← forward-looking pattern
  "the" in Head 2: focuses on "the" (49.5%)    ← self-reference pattern

Concatenated shape:  (3, 4)  ← 2 × 2 = 4 = d_model ✓
Final output after W_O: (3, 4)  ← back to (seq_len, d_model) ✓

Parameter Count:
  Single-Head: 3 × (4 × 4) = 48 params, 1 attention pattern
  Multi-Head:  2 × 3 × (4 × 2) + (4 × 4) = 64 params, 2 attention patterns
```
**Key observation:** Head 1 and Head 2 produce completely different attention patterns from the same input — this is the power of multi-head attention.

---
---

# Session 1.5: Feed-Forward Networks, Layer Normalization & Residual Connections
> **Revision Notes** — the other half of the Transformer block. Completing the full picture.

---

## Part 1: The Complete Transformer Block — Big Picture

A single Transformer block has **two sub-blocks**, each with a residual connection and normalization:

```
Input X (seq_len, d_model)
    │
    ├──────────────────┐
    ▼                  │  (residual)
  RMSNorm              │
    ▼                  │
  Multi-Head Attention │
    ▼                  │
  + ◄──────────────────┘  ← Add residual
    │
    ├──────────────────┐
    ▼                  │  (residual)
  RMSNorm              │
    ▼                  │
  Feed-Forward Network │
    ▼                  │
  + ◄──────────────────┘  ← Add residual
    │
Output (seq_len, d_model)  ← SAME shape as input!
```

A full Transformer = **N copies of this block stacked**:
- GPT-2: 12 blocks, Llama 3 8B: 32 blocks, GPT-3: 96 blocks

---

## Part 2: Feed-Forward Network (FFN)

### What It Is
A 2-layer neural network applied **independently** to each token position:
```
FFN(x) = Activation(x · W₁ + b₁) · W₂ + b₂
```

### The Expand → Activate → Contract Pattern
```
Input:  (seq_len, d_model)     e.g., (100, 4096)
     W₁: (4096, 16384)         ← EXPAND to 4× wider
     Activation (GELU/SwiGLU)  ← non-linearity
     W₂: (16384, 4096)         ← CONTRACT back to d_model
Output: (100, 4096)            ← same shape as input
```

The 4× expansion gives the network a **higher-dimensional "scratch pad"** to work in before compressing back.

### The Key Insight: FFN = Where Knowledge Is Stored

| Component | Role |
|-----------|------|
| **Multi-Head Attention** | Decides *which tokens to combine* (relationships) |
| **FFN** | Decides *what to do* with combined information (knowledge/transforms) |

Research shows FFN layers act as **key-value memories**:
- `W₁` acts like keys (pattern matching)
- `W₂` acts like values (stored knowledge)
- FFN contains **~63% of all parameters** in a Transformer layer!

### Activation Functions: ReLU → GELU → SwiGLU

```
Input:  [-2.0, -1.0, -0.5,  0.0,  0.5,  1.0,  2.0]
ReLU:   [ 0.0,  0.0,  0.0,  0.0,  0.5,  1.0,  2.0]   ← hard zero for negatives
GELU:   [-0.05,-0.16,-0.15, 0.0,  0.35, 0.84, 1.96]   ← smooth, small negatives pass
Swish:  [-0.24,-0.27,-0.19, 0.0,  0.31, 0.73, 1.76]   ← basis for SwiGLU
```

| Activation | Used By | Note |
|-----------|---------|------|
| **ReLU** | Original Transformer (2017) | Simple but loses info for negatives |
| **GELU** | GPT-2, GPT-3, BERT | Smooth approximation of ReLU |
| **SwiGLU** ✅ | Llama, Mistral, Gemma | Gating mechanism, current standard |

**SwiGLU formula:** `SwiGLU(x) = (Swish(x·W₁) ⊙ (x·W₃)) · W₂` — uses element-wise multiply of two projections (the "gating").

---

## Part 3: Residual Connections (Skip Connections)

### The Problem
Without residual connections, a 96-layer Transformer is **impossible to train** — gradients vanish, information from early layers gets washed out.

### The Solution: Just Add the Input Back
```
Output = Layer(x) + x      ← that's it! Just addition.
```

### Why It Works
1. **Gradient flow:** Even if `Layer(x)` has tiny gradients, the `+ x` path has gradient = 1 (identity) — learning never stops
2. **Incremental learning:** Layer only learns **what to add** (the residual), not the entire transformation from scratch
3. **Information preservation:** Input info is never lost — it's literally added back

### Demo
```
Input x:       [ 0.20,  1.42, -0.96,  1.94]
Layer(x):      [-0.17, -0.22,  0.58,  0.35]   ← transformation
Layer(x) + x:  [ 0.03,  1.20, -0.38,  2.29]   ← original info preserved!
```

> **Interview one-liner:** *"Residual connections let each layer learn what to ADD to the representation, rather than computing it from scratch. This prevents gradient vanishing and enables training 96+ layer networks."*

---

## Part 4: Layer Normalization

### What It Does
Normalizes activations across the feature dimension (d_model) for each token independently:
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

Where:
  μ = mean across d_model    (center)
  σ² = variance across d_model
  γ, β = learned parameters  (scale & shift)
```

### Pre-Norm vs. Post-Norm

| Aspect | Post-Norm (2017) | Pre-Norm (Modern) |
|--------|-------------------|-------------------|
| Norm position | After residual add | Before attention/FFN |
| Training stability | Harder (needs warmup) | Easier (more stable) |
| Used by | Original Transformer, BERT | GPT-2/3, Llama, Mistral |
| Residual path | Modified by norm | Clean identity ✅ |

**Pre-Norm wins** because the residual path (`+ x`) is completely unmodified — normalization happens *inside* the branch only.

### RMSNorm — The Current Standard

```
RMSNorm(x) = x / √(mean(x²) + ε) · γ

vs LayerNorm:
  ✗ No mean subtraction (no centering)
  ✗ No β parameter (no shift)
  ✓ Simpler and faster
  ✓ Same empirical quality
```

> **Interview answer:** *"Llama uses RMSNorm with Pre-Norm placement — it's simpler than LayerNorm (no centering), faster, and more training-stable."*

### Demo: LayerNorm vs RMSNorm
```
Input "the":  [ 0.20,  1.42, -0.96,  1.94]  (mean=0.650, var=1.263)

After LayerNorm: [-0.40, 0.69, -1.43, 1.15]  (mean≈0, var≈1)  ← centered + scaled
After RMSNorm:   [ 0.15, 1.09, -0.74, 1.49]  (RMS≈1)          ← scaled only
```

---

## Part 5: Parameter Count — Where Do All the Parameters Go?

### One Transformer Block (Llama 3 8B: d_model=4096, d_ff=14336, h=32)

| Component | Params | Share |
|-----------|--------|-------|
| MHA (W_Q + W_K + W_V + W_O) | ~67M | ~37% |
| FFN (W₁ + W₂) | ~117M | ~63% |
| Norms (γ only) | ~8K | <0.01% |
| **Per Layer Total** | **~184M** | 100% |

**FFN dominates!** It stores the model's "knowledge" — this is why larger models know more.

### Full Model
```
32 layers × ~184M = ~5.9B params (layer weights)
+ Embeddings (~524M)
+ Final output head
≈ 8B total parameters
```

---

## Part 6: The Full GPT Model (End-to-End)

```
Text Input: "The cat sat"
      │
  Tokenizer                          ← Session 1.1
      │
  Token IDs: [2, 5, 7]
      │
  Embedding Table                    ← Session 1.2
      │
  + Positional Encoding              ← Session 1.2
      │
  ┌─────────────────────┐
  │  Transformer Block 1 │           ← Sessions 1.3-1.5
  │  (MHA + FFN + Norms)│
  ├─────────────────────┤
  │  Transformer Block 2 │
  ├─────────────────────┤
  │       ...            │
  ├─────────────────────┤
  │  Transformer Block N │
  └─────────────────────┘
      │
  Final RMSNorm
      │
  Linear (d_model → vocab_size)      ← produces logits
      │
  Softmax → next token probabilities
```

---

## Quick Interview Reference Card — FFN, Norms & Residuals

| Concept | Key Answer |
|---------|-----------|
| What does FFN do? | 2-layer network (expand 4× → activate → contract). Applied per-token independently. Stores "knowledge" |
| FFN vs Attention role? | Attention = which tokens to combine. FFN = what to do with combined info |
| Why 4× expansion? | Larger computation space ("scratch pad"). Empirically 4× works well |
| SwiGLU? | Gating mechanism: `Swish(xW₁) ⊙ (xW₃) · W₂`. Current standard (Llama, Mistral) |
| What are residual connections? | `output = Layer(x) + x`. Prevents gradient vanishing, enables deep networks |
| Pre-Norm vs Post-Norm? | Pre-Norm = norm before layer (modern standard). Post-Norm = norm after (original paper) |
| RMSNorm vs LayerNorm? | RMSNorm = no centering, no β, faster. Same quality. Used by Llama/Mistral |
| Where are most params? | FFN (~63% per layer). FFN stores the model's learned knowledge |
| Why can blocks stack? | Input shape == Output shape: `(seq_len, d_model)`. Blocks are identical |

---

## Code References
- `mini_gpt/transformer_block.py` — Complete Transformer block with FFN, RMSNorm, residual connections, activation comparison, and step-by-step trace

### transformer_block.py Key Output
```text
Architecture: x → RMSNorm → MHA → +x → RMSNorm → FFN → +x

Step-by-step trace for "the cat sat" (d_model=4, d_ff=16, h=2):
  0. Input:         "the": [ 0.20,  1.42, -0.96,  1.94]
  1. After RMSNorm: "the": [ 0.15,  1.09, -0.74,  1.49]
  2. After MHA:     "the": [-0.19,  0.22, -0.18, -0.04]
  3. After +x:      "the": [ 0.01,  1.64, -1.14,  1.90]  ← residual preserves input
  4. After RMSNorm: "the": [ 0.01,  1.19, -0.83,  1.38]
  5. After FFN:     "the": [-0.07,  0.05, -0.19,  0.02]
  6. After +x:      "the": [-0.06,  1.69, -1.33,  1.93]  ← final output

Input shape == Output shape: (3, 4) == (3, 4) ✅
FFN share: 67.3% of block parameters ← FFN dominates!
```

---
---

# Session 1.6: Encoder, Decoder & Encoder-Decoder Architectures
> **Revision Notes** — the three Transformer variants, three types of attention, autoregressive generation, and KV-cache. Heavily tested in interviews.

---

## Part 1: Three Architectures from One Building Block

```
Original Transformer (2017):     [Encoder] ──cross-attention──► [Decoder]

Three variants emerged:
  1. Encoder-only:    [Encoder]              → BERT
  2. Decoder-only:    [Decoder]              → GPT, Llama, Mistral
  3. Encoder-Decoder: [Encoder] → [Decoder]  → T5, BART
```

---

## Part 2: Encoder-Only (BERT)

**Bidirectional** self-attention — every token sees every other token (no mask).

```
Input: "The cat [MASK] on the mat"
         │
  Transformer Blocks (×12)     ← bidirectional attention
         │
  Contextualized Representations
         │
  Task-specific head (classification, NER, QA)
```

### Key Properties
- **Sees full context**: "bank" gets different representations in "river bank" vs "bank account"
- **Cannot generate**: Because it sees the future, can't do autoregressive generation
- **Pre-training**: Masked Language Modeling (MLM) — mask 15% of tokens, predict them
- **Use cases**: Classification, sentiment, NER, QA, search/retrieval

### Example Models
| Model | Params | Note |
|-------|--------|------|
| BERT-Base | 110M | 12 layers, 768 d_model |
| BERT-Large | 340M | 24 layers, 1024 d_model |
| RoBERTa | 125M-355M | Better training recipe |
| DeBERTa | 140M-400M | Disentangled attention |

---

## Part 3: Decoder-Only (GPT, Llama) — The Dominant Architecture

**Causal** self-attention — each token can only see itself and preceding tokens.

```
Input: "The cat sat"
         │
  Transformer Blocks (×32)     ← causal mask (lower-triangular)
         │
  Linear (d_model → vocab_size) → Softmax → predict NEXT token
```

### Autoregressive Generation — Step by Step

```
Step 1: Input ["The"]           → Model predicts "cat"   (90.6%)
Step 2: Input ["The", "cat"]    → Model predicts "sat"   (94.6%)
Step 3: Input ["The", "cat", "sat"] → Model predicts "on" (93.7%)
...continues until <EOS> or max length
```

The model generates **one token at a time**, feeding previously generated tokens back in.

### Training vs. Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Processing | Full sequence in parallel | One token at a time |
| Trick | Causal mask prevents peeking | KV-cache avoids recomputation |
| Supervision | Teacher forcing (real next tokens) | Sample from predictions |

### KV-Cache — Critical Optimization

**Without KV-cache:** Generating token N recomputes K,V for ALL N tokens → O(N²) total.
**With KV-cache:** Only compute K,V for the NEW token, use cached K,V for past tokens → O(N) total.

```
Without cache: Step 1→1 token, Step 2→2 tokens, Step 3→3 tokens = 1+2+3+4 = O(N²)
With cache:    Step 1→1 new,   Step 2→1 new,   Step 3→1 new    = 1+1+1+1 = O(N)
```

**Trade-off:** KV-cache uses significant memory:
| Context Length | KV-Cache Size (Llama 3 70B) |
|---------------|-----------------------------|
| 1,024 tokens | 2.5 GB |
| 4,096 tokens | 10 GB |
| 32,768 tokens | 80 GB |
| 131,072 tokens | 320 GB |

> **Interview answer:** *"KV-cache stores the Key and Value matrices from previous tokens during generation, so each new token only computes its own Q,K,V rather than reprocessing the entire sequence. This reduces per-step complexity from O(N²) to O(N), but at the cost of GPU memory."*

### Example Models
| Model | Layers | d_model | Params |
|-------|--------|---------|--------|
| GPT-2 | 12-48 | 768-1600 | 117M-1.5B |
| GPT-3 | 96 | 12,288 | 175B |
| Llama 3 8B | 32 | 4,096 | 8B |
| Llama 3 70B | 80 | 8,192 | 70B |
| Mistral 7B | 32 | 4,096 | 7B |

---

## Part 4: Encoder-Decoder (T5, BART)

Two stacks: **encoder** (bidirectional) processes input, **decoder** (causal) generates output, connected by **cross-attention**.

```
Input: "Translate: The cat sat"
         │
  ┌─── ENCODER ───┐
  │ Bidirectional  │     ← sees full input
  └───────┬────────┘
          │ encoder representations
  ┌─── DECODER ───┐
  │ 1. Causal Self-Attn │   ← sees only past output
  │ 2. Cross-Attention  │   ← looks at encoder output
  │ 3. FFN              │
  └───────┬────────┘
          │
  "Le chat s'est assis"
```

### Cross-Attention — The Bridge

```
Q comes from DECODER  ("what output token am I generating?")
K, V come from ENCODER ("what does the input contain?")

Cross-Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec · K_enc^T / √d_k) · V_enc
```

### Example Models
| Model | Params | Best For |
|-------|--------|----------|
| T5 | 220M-11B | Translation, summarization, QA |
| BART | 140M-400M | Summarization, denoising |
| Flan-T5 | 80M-11B | Instruction-following |

---

## Part 5: Three Types of Attention — Master Table

| Attention Type | Q, K, V Source | Mask | Used Where |
|---------------|----------------|------|------------|
| **Bidirectional Self-Attention** | All from same input | None (full) | Encoder (BERT) |
| **Causal Self-Attention** | All from same input | Lower-triangular | Decoder (GPT, Llama) |
| **Cross-Attention** | Q from decoder, K/V from encoder | None | Encoder-Decoder bridge (T5) |

---

## Part 6: Why Decoder-Only Won

| Factor | Encoder-Only | Encoder-Decoder | Decoder-Only ✅ |
|--------|-------------|-----------------|----------------|
| Text generation | ❌ No | ✅ Yes | ✅ Yes |
| Understanding | ✅ Excellent | ✅ Good | ✅ Good (with scale) |
| Simplicity | Medium | Complex (2 stacks) | Simple (1 stack) |
| Scaling | Plateaus ~1B | Good but complex | **Scales best** |
| Pre-training signal | MLM (15% of tokens) | Span corruption | **Next-token (100%)** |
| Versatility | Task-specific heads | Multi-task | **Any task via prompting** |

> **Interview answer:** *"Decoder-only dominates because (1) next-token prediction uses 100% of tokens as training signal vs 15% for MLM, (2) it scales predictably with compute/data, and (3) one architecture handles ALL tasks via prompting — no task-specific heads needed."*

---

## Quick Interview Reference Card — Architectures

| Concept | Key Answer |
|---------|-----------|
| Encoder-only example? | BERT — bidirectional attention, MLM pre-training, classification/NER/search |
| Decoder-only example? | GPT, Llama — causal attention, next-token prediction, generation |
| Encoder-decoder example? | T5, BART — bidirectional encoder + causal decoder + cross-attention, translation |
| Three attention types? | Bidirectional (encoder), Causal (decoder), Cross (encoder→decoder bridge) |
| What is cross-attention? | Q from decoder, K/V from encoder — lets decoder look at the input |
| What is autoregressive? | Generate one token at a time, feed it back in as input for next prediction |
| What is KV-cache? | Cache K,V matrices from past tokens during generation → O(N²) → O(N) per step |
| KV-cache trade-off? | Saves compute but uses memory (Llama 70B at 128K context ≈ 320GB cache) |
| Why decoder-only won? | 100% training signal, scales best, one architecture for all tasks via prompting |

---

## Code References
- `mini_gpt/architectures.py` — Side-by-side comparison of bidirectional, causal, and cross-attention + autoregressive generation demo + KV-cache analysis

### architectures.py Key Output
```text
1. Bidirectional (BERT):     "the" sees all tokens equally
2. Causal (GPT):             "the" sees only itself (100%)
3. Cross-Attention (T5):     "le" (French) attends to "the","cat","sat" (English)

Autoregressive Generation:
  Step 1: [<BOS>]              → "the"  (90.6%)
  Step 2: [<BOS> the]          → "cat"  (93.1%)
  Step 3: [<BOS> the cat]      → "sat"  (94.6%)
  ...
  Final: <BOS> the cat sat on mat <EOS>

KV-Cache: Llama 3 70B at 131K context = 320 GB of cache memory
```

---
---

# Session 1.7: LLM Training Pipeline — Pre-training → SFT → RLHF/DPO
> **Revision Notes** — the three-stage pipeline that turns a random Transformer into ChatGPT. Critical interview topic.

---

## Part 1: The Three-Stage Pipeline — Overview

```
Stage 1: PRE-TRAINING              Stage 2: SFT                    Stage 3: RLHF / DPO
"Learn language itself"            "Learn to follow instructions"   "Learn human preferences"

Next-token prediction              Mimic good examples              Prefer good over bad
Trillions of tokens                ~100K instruction pairs          ~50K-100K comparisons
Months, $10M-$100M+               Hours-Days, $1K-$100K            Days, $10K-$1M
→ Smart but unhelpful              → Helpful but may be wrong       → Helpful, harmless, honest
```

---

## Part 2: Pre-Training — "Learn Language Itself"

### The Objective: Next-Token Prediction
```
Input:    [The] [cat]  [sat]  [on]  [the]
Target:   [cat] [sat]  [on]  [the]  [mat]

Loss = CrossEntropy averaged across ALL positions
→ From 5 tokens, you get 5 training examples. MLM only uses 15%.
```

### Scale of Pre-Training

| Model | Training Tokens | Estimated Cost |
|-------|----------------|---------------|
| GPT-3 (175B) | 300B | ~$5M |
| Llama 2 70B | 2T | ~$2M |
| Llama 3 405B | 15T | ~$100M+ |

### What Pre-Training Learns
- **Syntax & grammar**: How language works
- **World knowledge**: Facts encoded implicitly through prediction patterns
- **Reasoning**: Logical patterns from structured text
- **Code**: Syntax, algorithms, patterns

### Scaling Laws (Chinchilla)
```
Optimal tokens ≈ 20 × number of parameters
10B params → ~200B tokens
70B params → ~1.4T tokens
```

> **Interview insight:** *"Chinchilla showed many early LLMs were under-trained — GPT-3 (175B) trained on 300B tokens when optimal was ~3.5T. Llama was designed 'compute-optimal' by training smaller models on more data."*

---

## Part 3: Supervised Fine-Tuning (SFT)

### The Problem: Base Models Don't Answer Questions
```
Base model:  "What is the capital of France?"
→ "What is the capital of Germany? What is the capital of Italy?"
   ← Just continues text! Doesn't answer.

After SFT:   "What is the capital of France?"
→ "The capital of France is Paris."
   ← Actually answers!
```

### How SFT Works
Fine-tune on instruction-response pairs:
```json
{"instruction": "What is the capital of France?",
 "response": "The capital of France is Paris."}
```

### The Loss Masking Trick
```
Prompt tokens:   [What] [is] [the] [capital] [?]  → loss = 0 (masked)
Response tokens: [The] [capital] [is] [Paris] [.]  → loss computed here only
```
Only backpropagate on response tokens — don't penalize the model for "getting the prompt wrong."

### Key Details
- **Data size**: 10K-100K high-quality examples
- **Quality > quantity**: 1,000 excellent > 100,000 mediocre
- **Duration**: Hours to days
- **What it teaches**: The FORMAT of being helpful, not new knowledge

> **Interview insight:** *"SFT teaches the model the format of being helpful — how to structure answers, when to refuse. The knowledge itself comes from pre-training."*

---

## Part 4: RLHF — Reinforcement Learning from Human Feedback

### The Problem with SFT
SFT models may still be harmful, overconfident, verbose, or fail to refuse dangerous requests. SFT teaches "what to say" but not "what's **better** to say."

### The Three Steps

#### Step 1: Collect Human Comparisons
Humans rank model responses:
```
Prompt: "Explain gravity"
Response A: "Gravity is the fundamental force..."  ← Human picks A > B
Response B: "lol gravity is when stuff falls"
```

#### Step 2: Train Reward Model (RM)
Separate model that scores responses:
```
RM("Explain gravity", "Gravity is the fundamental force...") → 0.85
RM("Explain gravity", "lol gravity is when stuff falls")     → 0.31
```

#### Step 3: PPO Optimization
Update the LLM using RM as a judge:
```
Objective = maximize RM(prompt, response) - β × KL(policy ∥ SFT_model)
            ↑ "be good"                      ↑ "don't drift too far"
```

### Why KL Penalty Matters
Without it, the model "hacks" the reward model — finds degenerate outputs that score high but are nonsensical. KL keeps it close to the SFT model.

---

## Part 5: DPO — Direct Preference Optimization (The Simpler Alternative)

### The Problem with RLHF
- 3 models to train (SFT + RM + PPO)
- PPO is notoriously unstable
- RM can be "hacked"

### DPO's Insight: Skip the Reward Model
```
RLHF:  Comparisons → Train RM → PPO with RM → Update LLM  (3 models)
DPO:   Comparisons → Directly update LLM                   (1 model)
```

DPO directly increases the probability of preferred responses and decreases rejected ones:
```
"Increase P(good response) relative to reference model,
 decrease P(bad response) relative to reference model."
```

### RLHF vs DPO

| Aspect | RLHF | DPO |
|--------|------|-----|
| Models to train | 3 (SFT + RM + PPO) | 1 (direct) |
| Stability | Hard, unstable | Stable |
| Compute | High | Lower |
| Quality | Slightly better at scale | Comparable |
| Used by | Early ChatGPT, Claude 1 | Llama 2/3, Zephyr, open-source |

> **Interview answer:** *"DPO eliminates the reward model and PPO loop by directly optimizing on preference pairs. The key insight: the optimal RL policy can be expressed in closed form, so you don't need RL at all."*

---

## Part 6: Complete Pipeline End-to-End

```
┌─────────────────────────────────────────────────────────┐
│  PRE-TRAINING                                           │
│  15T tokens, next-token prediction                      │
│  Result: "Smart text completer"                         │
├─────────────────────────────────────────────────────────┤
│  SFT (Supervised Fine-Tuning)                           │
│  ~100K instruction pairs                                │
│  Result: "Helpful assistant"                            │
├─────────────────────────────────────────────────────────┤
│  ALIGNMENT (RLHF or DPO)                                │
│  ~50K-100K preference comparisons                       │
│  Result: "Helpful, harmless, honest"                    │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Interview Reference Card — Training Pipeline

| Concept | Key Answer |
|---------|-----------|
| Pre-training objective? | Next-token prediction across trillions of tokens |
| Why more sample-efficient than MLM? | Every token is a training target (100% vs 15% for MLM) |
| Chinchilla scaling law? | Optimal tokens ≈ 20× model parameters |
| What does SFT teach? | The FORMAT of being helpful (not new knowledge — that's from pre-training) |
| SFT loss masking? | Only compute loss on response tokens, not prompt tokens |
| RLHF pipeline? | Collect human comparisons → Train Reward Model → PPO optimization |
| Why KL penalty in RLHF? | Prevents reward hacking — keeps model close to SFT baseline |
| What is DPO? | Direct Preference Optimization — skips RM and PPO, directly optimizes on preferences |
| RLHF vs DPO? | DPO is simpler (1 model vs 3), more stable, comparable quality |
| Pipeline order? | Pre-train → SFT → RLHF/DPO (must be in this order) |

---
---

# Session 1.8: Generation Parameters — Temperature, Top-k, Top-p
> **Revision Notes** — how to control the randomness, quality, and diversity of LLM outputs.

---

## Part 1: The Sampling Pipeline

```
Model output (logits):  [5.0, 3.5, 2.0, 0.5, ...]   (vocab_size numbers)
         │
    ① Temperature scaling  (logits / T)
         │
    ② Top-k / Top-p filtering
         │
    ③ Softmax → probabilities
         │
    ④ Sample → next token
```

---

## Part 2: Temperature

Divides logits before softmax: `adjusted_logits = logits / T`

| Temperature | Effect | Use Case |
|------------|--------|----------|
| T ≈ 0 | Greedy/deterministic — always picks most likely | Code, math, factual Q&A |
| T = 0.5 | Focused — slight variation | Reliable with minor creativity |
| T = 0.7 | Balanced | General conversation |
| T = 1.0 | Original learned distribution | Baseline |
| T = 1.5-2.0 | Very flat — near-random | Brainstorming, creative writing |

### Worked Example
```
Logits: [5.0, 3.5, 2.0] for tokens: "cat", "dog", "fish"

T=0.5: softmax → cat(95.0%), dog(4.7%), fish(0.2%)   ← almost always "cat"
T=1.0: softmax → cat(77.4%), dog(17.3%), fish(3.9%)  ← usually "cat"
T=2.0: softmax → cat(50.9%), dog(24.1%), fish(11.4%) ← anything possible
```

> **One-liner:** *"Temperature controls randomness — lower = more deterministic, higher = more creative."*

---

## Part 3: Top-k Sampling

Keep only the **k most probable** tokens, zero out everything else, renormalize.

```
Top-k=3: Keep cat(78.6%), dog(17.5%), fish(3.9%). Remove all others.
```

**Problem:** k is FIXED — doesn't adapt to confidence. When model is 99% sure, k=10 includes 9 noise tokens. When model is uncertain, k=3 throws away valid options.

---

## Part 4: Top-p (Nucleus Sampling) — The Better Alternative

Keep the **smallest set** of tokens whose cumulative probability ≥ p.

```
Top-p=0.9 when confident:  Keeps 1 token (just "cat" at 95%)   ← tight!
Top-p=0.9 when uncertain:  Keeps 7 tokens (many at ~15% each)  ← wide!
```

**Top-p adapts** — tight when confident, wide when uncertain. Strictly better than top-k.

| p Value | Effect |
|---------|--------|
| p = 0.5 | Very conservative |
| p = 0.9 | Standard (GPT-3 default) |
| p = 1.0 | No filtering |

---

## Part 5: Frequency & Presence Penalties

Prevent repetition:
```
adjusted_logits[i] = logits[i] - freq_penalty × count(token_i)
                                - presence_penalty × (1 if token_i appeared else 0)
```

| Parameter | Effect |
|-----------|--------|
| **freq_penalty** | Penalizes proportional to count — reduces "the the the" loops |
| **presence_penalty** | Penalizes if appeared at all — encourages topic diversity |

---

## Part 6: Recommended Settings

| Use Case | Temperature | Top-p | Notes |
|----------|------------|-------|-------|
| Code generation | 0 | 0.95 | Deterministic |
| Factual Q&A | 0-0.3 | 0.9 | Very reliable |
| General chat | 0.7 | 0.9 | Balanced |
| Creative writing | 0.9-1.2 | 0.95 | Expressive |
| Brainstorming | 1.0-1.5 | 1.0 | Maximum diversity |
| **Agents/production** | **0** | **0.95** | **Reliability > creativity** |

---

## Quick Interview Reference Card — Generation Parameters

| Concept | Key Answer |
|---------|-----------|
| What is temperature? | Divides logits before softmax. Lower = deterministic, higher = random |
| T=0 vs T=1? | T=0 is greedy (always picks most likely). T=1 uses learned distribution as-is |
| What is top-k? | Keep only k most probable tokens, zero rest. Fixed k doesn't adapt to confidence |
| What is top-p? | Keep smallest token set with cumulative prob ≥ p. Adapts to confidence |
| Top-k vs top-p? | Top-p is better — adaptive. Top-k is fixed and either too tight or too loose |
| Frequency penalty? | Penalizes tokens proportional to how many times they've appeared (reduces repetition) |
| Presence penalty? | Penalizes any token that appeared at all (encourages diversity) |
| Best for code? | T=0 (greedy), top-p=0.95 |
| Best for chat? | T=0.7, top-p=0.9 |

---

## Code References
- `mini_gpt/generation_params.py` — Interactive demo of temperature, top-k, top-p, penalties with visual bar charts and side-by-side comparisons

### generation_params.py Key Output
```text
Temperature comparison:
  T=0.1: cat=100% (greedy)
  T=0.5: cat=95%, dog=5%
  T=1.0: cat=77%, dog=17%, fish=4%
  T=2.0: cat=51%, dog=24%, fish=11%

Top-p adapts to confidence:
  Confident model: Top-p=0.9 keeps 1 token (tight!)
  Uncertain model: Top-p=0.9 keeps 7 tokens (wide!)
```
