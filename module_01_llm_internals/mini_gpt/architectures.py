"""
Session 1.6 — Encoder, Decoder & Encoder-Decoder Architectures (NumPy)
======================================================================
Demonstrates the THREE types of attention and architectures side by side:
  1. Bidirectional Self-Attention (Encoder / BERT-style)
  2. Causal Self-Attention (Decoder / GPT-style)
  3. Cross-Attention (Encoder-Decoder / T5-style)

Also shows:
  4. Autoregressive generation loop
  5. KV-Cache optimization
"""

import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores)
    return weights @ V, weights


def create_causal_mask(seq_len):
    """Lower-triangular mask for autoregressive attention."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def print_weights(weights, row_labels, col_labels, title=""):
    """Pretty-print attention weights."""
    if title:
        print(f"\n  {title}:")
    header = "           " + "  ".join(f"{l:>7s}" for l in col_labels)
    print(header)
    for i in range(len(row_labels)):
        row = "  ".join(f"{weights[i, j]:7.3f}" for j in range(len(col_labels)))
        print(f'  {row_labels[i]:>7s}  {row}')


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    d_model = 4
    
    print("=" * 70)
    print("  SESSION 1.6 — THREE ARCHITECTURES COMPARED")
    print("=" * 70)
    
    # ── Input for encoder-side demos ──
    X = np.array([
        [ 0.20,  1.42, -0.96,  1.94],   # "the"
        [ 1.07, -0.18, -0.41,  0.73],   # "cat"
        [ 1.09, -1.32,  0.24,  0.34],   # "sat"
    ])
    enc_labels = ['"the"', '"cat"', '"sat"']
    
    # Shared weight matrices for comparison
    W_Q = np.random.randn(d_model, d_model) * 0.3
    W_K = np.random.randn(d_model, d_model) * 0.3
    W_V = np.random.randn(d_model, d_model) * 0.3
    
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    # ══════════════════════════════════════════════════════
    #  1. BIDIRECTIONAL SELF-ATTENTION (Encoder / BERT)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  1. BIDIRECTIONAL SELF-ATTENTION (Encoder / BERT-style)")
    print(f"{'='*70}")
    print('  Every token sees EVERY other token — no mask.')
    
    _, bidir_weights = attention(Q, K, V, mask=None)
    print_weights(bidir_weights, enc_labels, enc_labels,
                  "Attention Weights (bidirectional — no mask)")
    
    print("""
  → "the" can see "cat" AND "sat" (including future!)
  → Used for: BERT, RoBERTa — classification, NER, search
  → ❌ Cannot do text generation (sees the future)""")
    
    # ══════════════════════════════════════════════════════
    #  2. CAUSAL SELF-ATTENTION (Decoder / GPT)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  2. CAUSAL SELF-ATTENTION (Decoder / GPT-style)")
    print(f"{'='*70}")
    print('  Each token can only see itself and previous tokens.')
    
    causal_mask = create_causal_mask(3)
    _, causal_weights = attention(Q, K, V, mask=causal_mask)
    print_weights(causal_weights, enc_labels, enc_labels,
                  "Attention Weights (causal — lower-triangular mask)")
    
    print(f"\n  Causal Mask:")
    for i in range(3):
        row = "  ".join("✓" if causal_mask[i,j] else "✗" for j in range(3))
        print(f"    {enc_labels[i]:>5s}: {row}")
    
    print("""
  → "the" can ONLY see itself (100% self-attention)
  → "cat" sees "the" + itself, but NOT "sat"
  → Used for: GPT, Llama, Mistral — text generation
  → ✅ Enables autoregressive generation""")
    
    # ══════════════════════════════════════════════════════
    #  3. CROSS-ATTENTION (Encoder-Decoder / T5)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  3. CROSS-ATTENTION (Encoder-Decoder / T5-style)")
    print(f"{'='*70}")
    print('  Decoder queries attend to encoder key-values.')
    print('  Q comes from DECODER, K & V come from ENCODER.')
    
    # Simulate decoder tokens (e.g., French translation)
    X_dec = np.array([
        [ 0.50,  0.30, -0.10,  0.80],   # "le"
        [ 0.90, -0.40,  0.20,  0.60],   # "chat"
    ])
    dec_labels = ['"le"', '"chat"']
    
    W_Q_dec = np.random.randn(d_model, d_model) * 0.3
    
    Q_dec = X_dec @ W_Q_dec    # Q from DECODER
    K_enc = K                  # K from ENCODER (computed earlier)
    V_enc = V                  # V from ENCODER
    
    print(f"\n  Encoder input: {enc_labels}  (English: 'the cat sat')")
    print(f"  Decoder input: {dec_labels}  (French: 'le chat ...')")
    print(f"\n  Q shape: {Q_dec.shape} (from decoder — {len(dec_labels)} tokens)")
    print(f"  K shape: {K_enc.shape} (from encoder — {len(enc_labels)} tokens)")
    print(f"  V shape: {V_enc.shape} (from encoder — {len(enc_labels)} tokens)")
    
    _, cross_weights = attention(Q_dec, K_enc, V_enc, mask=None)
    print_weights(cross_weights, dec_labels, enc_labels,
                  "Cross-Attention Weights (decoder Q → encoder K,V)")
    
    print("""
  → "le" (French for "the") attends most to encoder tokens
  → "chat" (French for "cat") attends most to its English equivalent
  → This is how translation models align source → target language
  → Used for: T5, BART, mBART — translation, summarization""")
    
    # ══════════════════════════════════════════════════════
    #  4. AUTOREGRESSIVE GENERATION DEMO
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  4. AUTOREGRESSIVE GENERATION (Decoder-only)")
    print(f"{'='*70}")
    
    # Simulate a tiny vocabulary
    vocab = {0: "<BOS>", 1: "the", 2: "cat", 3: "sat", 4: "on", 5: "mat", 6: "<EOS>"}
    vocab_size = len(vocab)
    
    # Simulated logits for each step (normally these come from the model)
    # Each row is logits over vocab — we'll make them favor the right next token
    simulated_logits = [
        np.array([-1, 3.5, 0.2, -0.5, -1, -0.3, -2]),   # after <BOS> → "the"
        np.array([-1, -0.5, 3.8, 0.1, -0.3, -1, -2]),    # after "the" → "cat"
        np.array([-1, -0.3, -0.5, 4.1, 0.2, -1, -2]),    # after "the cat" → "sat"
        np.array([-1, -0.5, -1, -0.3, 3.9, 0.1, -2]),    # after "the cat sat" → "on"
        np.array([-1, 0.2, -1, -0.5, -0.3, 4.2, -2]),    # after "... sat on" → "mat"
        np.array([-2, -1, -1, -1, -1, -1, 4.5]),          # after "... on mat" → <EOS>
    ]
    
    print(f"\n  Vocabulary: {vocab}")
    print(f"\n  Generation trace (greedy — pick highest probability):\n")
    
    tokens = [0]  # Start with <BOS>
    for step in range(len(simulated_logits)):
        logits = simulated_logits[step]
        probs = softmax(logits.reshape(1, -1)).flatten()
        next_token = np.argmax(probs)
        
        current_text = " ".join(vocab[t] for t in tokens)
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = ", ".join(f"{vocab[i]}({probs[i]:.1%})" for i in top3_idx)
        
        print(f"  Step {step+1}: [{current_text}]")
        print(f"          Top-3 predictions: {top3}")
        print(f"          Selected: \"{vocab[next_token]}\"")
        
        tokens.append(next_token)
        
        if next_token == 6:  # <EOS>
            break
    
    final_text = " ".join(vocab[t] for t in tokens)
    print(f"\n  ✅ Final output: {final_text}")
    
    # ══════════════════════════════════════════════════════
    #  5. KV-CACHE DEMO
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  5. KV-CACHE — Why It Matters")
    print(f"{'='*70}")
    
    print(f"""
  WITHOUT KV-Cache (naive autoregressive):
  ┌────────────────────────────────────────────────────────────┐
  │ Step 1: Compute K,V for [The]                   → 1 token │
  │ Step 2: Compute K,V for [The, cat]              → 2 tokens│
  │ Step 3: Compute K,V for [The, cat, sat]         → 3 tokens│
  │ Step 4: Compute K,V for [The, cat, sat, on]     → 4 tokens│
  │ Total K,V computations: 1+2+3+4 = 10  ← O(N²)           │
  └────────────────────────────────────────────────────────────┘

  WITH KV-Cache (efficient):
  ┌────────────────────────────────────────────────────────────┐
  │ Step 1: Compute K,V for [The], CACHE             → 1 new  │
  │ Step 2: Compute K,V for [cat] only, append cache  → 1 new │
  │ Step 3: Compute K,V for [sat] only, append cache  → 1 new │
  │ Step 4: Compute K,V for [on]  only, append cache  → 1 new │
  │ Total K,V computations: 1+1+1+1 = 4   ← O(N)            │
  └────────────────────────────────────────────────────────────┘""")
    
    # Simulate KV-cache growth
    print(f"\n  KV-Cache memory for Llama 3 70B (d_model=8192, h=64, 80 layers):")
    d_model_real = 8192
    n_layers = 80
    for ctx_len in [1024, 4096, 32768, 131072]:
        # 2 (K+V) × n_layers × ctx_len × d_model × 2 bytes (FP16)
        cache_bytes = 2 * n_layers * ctx_len * d_model_real * 2
        cache_gb = cache_bytes / (1024**3)
        print(f"    Context {ctx_len:>7,} tokens: {cache_gb:>6.1f} GB")
    
    # ══════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  ARCHITECTURE COMPARISON")
    print(f"{'='*70}")
    print(f"""
  ┌──────────────┬─────────────────┬──────────────────┬─────────────────┐
  │              │ Encoder-Only    │ Decoder-Only     │ Encoder-Decoder │
  │              │ (BERT)          │ (GPT/Llama)      │ (T5/BART)       │
  ├──────────────┼─────────────────┼──────────────────┼─────────────────┤
  │ Attention    │ Bidirectional   │ Causal (masked)  │ Both + Cross    │
  │ Generation   │ ❌ No           │ ✅ Yes           │ ✅ Yes          │
  │ Understanding│ ✅ Excellent    │ ✅ Good (scale)  │ ✅ Good         │
  │ Pre-training │ MLM (15%)      │ Next-token (100%)│ Span corruption │
  │ Use cases    │ Classification │ General-purpose  │ Translation     │
  │              │ NER, Search    │ Chat, Code, Any  │ Summarization   │
  │ Scaling      │ Plateaus ~1B   │ Scales best      │ Good but complex│
  │ Dominance    │ 2018-2020      │ 2020-present ✅  │ Niche use cases │
  └──────────────┴─────────────────┴──────────────────┴─────────────────┘

  Why decoder-only won:
  1. Next-token prediction uses 100% of tokens as training signal (vs 15% for MLM)
  2. Scales predictably with more compute/data (scaling laws)
  3. One architecture handles ALL tasks via prompting — no task-specific heads
""")
