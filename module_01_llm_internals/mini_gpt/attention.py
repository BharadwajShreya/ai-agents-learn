"""
Session 1.3 — Self-Attention from Scratch (NumPy)
==================================================
Builds on embedding.py — we take the final embeddings from Session 1.2
and pass them through the full self-attention mechanism.

Covers:
  1. Creating Q, K, V with learned weight matrices
  2. Computing attention scores (Q · K^T)
  3. Scaling by √d_k
  4. Softmax to get attention weights
  5. Weighted sum of Values → context-enriched output
  6. Causal masking for autoregressive models (GPT-style)
"""

import numpy as np


def softmax(x):
    """Row-wise softmax: each row becomes a probability distribution summing to 1.
    
    We subtract the max per row for numerical stability (prevents overflow in exp).
    This doesn't change the result because softmax(x) = softmax(x - c) for any constant c.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def self_attention(X, W_Q, W_K, W_V, mask=None):
    """
    Full self-attention computation.
    
    Args:
        X:    Input embeddings     (seq_len, d_model)
        W_Q:  Query weight matrix  (d_model, d_k)
        W_K:  Key weight matrix    (d_model, d_k)
        W_V:  Value weight matrix  (d_model, d_v)
        mask: Optional causal mask (seq_len, seq_len) — True where we ALLOW attention
    
    Returns:
        output:    Context-enriched vectors  (seq_len, d_v)
        weights:   Attention weight matrix   (seq_len, seq_len)
    """
    # ── Step 1: Project input into Q, K, V ──
    Q = X @ W_Q    # (seq_len, d_k) — "What am I looking for?"
    K = X @ W_K    # (seq_len, d_k) — "What do I contain?"
    V = X @ W_V    # (seq_len, d_v) — "What information can I provide?"
    
    d_k = Q.shape[-1]
    
    # ── Step 2: Compute raw attention scores ──
    # Q · K^T gives us a (seq_len, seq_len) matrix
    # scores[i][j] = "how relevant is token j to token i?"
    scores = Q @ K.T
    
    # ── Step 3: Scale by √d_k ──
    # Without this, large d_k → large dot products → softmax saturation → dead gradients
    scaled_scores = scores / np.sqrt(d_k)
    
    # ── Step 4 (optional): Apply causal mask ──
    # Set future positions to -inf so softmax gives them 0 probability
    if mask is not None:
        scaled_scores = np.where(mask, scaled_scores, -np.inf)
    
    # ── Step 5: Softmax → attention weights ──
    # Each row becomes a probability distribution (sums to 1)
    attn_weights = softmax(scaled_scores)
    
    # ── Step 6: Weighted sum of Values ──
    # Each token's output = weighted combination of all Value vectors
    output = attn_weights @ V
    
    return output, attn_weights


def create_causal_mask(seq_len):
    """Create a lower-triangular mask for autoregressive (GPT-style) attention.
    
    Returns a boolean matrix where True = allowed to attend.
    
    For seq_len=3:
        [[ True, False, False],     ← token 0 sees only itself
         [ True,  True, False],     ← token 1 sees tokens 0-1
         [ True,  True,  True]]     ← token 2 sees tokens 0-2
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def print_attention_map(weights, labels):
    """Pretty-print the attention weight matrix with token labels."""
    n = len(labels)
    header = "         " + "  ".join(f"{l:>7s}" for l in labels)
    print(header)
    for i in range(n):
        row = "  ".join(f"{weights[i, j]:7.3f}" for j in range(n))
        print(f'{labels[i]:>7s}  {row}   (sum={weights[i].sum():.2f})')


# ============================================================
#  DEMO: Full Self-Attention on "the cat sat"
# ============================================================
if __name__ == "__main__":
    # ── Setup: Use the same final embeddings from Session 1.2 ──
    # These are Token Embedding + Positional Encoding from embedding.py
    X = np.array([
        [ 0.20,  1.42, -0.96,  1.94],   # "the" (position 0)
        [ 1.07, -0.18, -0.41,  0.73],   # "cat" (position 1)
        [ 1.09, -1.32,  0.24,  0.34],   # "sat" (position 2)
    ])
    
    seq_len, d_model = X.shape
    d_k = d_model   # For simplicity, d_k = d_model = 4
    labels = ['"the"', '"cat"', '"sat"']
    
    print("=" * 60)
    print("  SESSION 1.3 — SELF-ATTENTION FROM SCRATCH")
    print("=" * 60)
    
    print(f"\n📥 Input X (final embeddings from Session 1.2):")
    print(f"   Shape: ({seq_len}, {d_model}) = (seq_len, d_model)")
    for i, label in enumerate(labels):
        print(f"   {label:>5s}: {X[i].round(2)}")
    
    # ── Initialize weight matrices (random but fixed seed) ──
    np.random.seed(123)
    W_Q = np.random.randn(d_model, d_k) * 0.3
    W_K = np.random.randn(d_model, d_k) * 0.3
    W_V = np.random.randn(d_model, d_k) * 0.3
    
    # ── Step 1: Q, K, V ──
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    print(f"\n{'='*60}")
    print("  STEP 1: Create Q, K, V (X @ W_Q, X @ W_K, X @ W_V)")
    print(f"{'='*60}")
    print(f"\n  Q = X · W_Q  →  shape {Q.shape}")
    for i, label in enumerate(labels):
        print(f"   Q[{label:>5s}] = {Q[i].round(3)}")
    print(f"\n  K = X · W_K  →  shape {K.shape}")
    for i, label in enumerate(labels):
        print(f"   K[{label:>5s}] = {K[i].round(3)}")
    print(f"\n  V = X · W_V  →  shape {V.shape}")
    for i, label in enumerate(labels):
        print(f"   V[{label:>5s}] = {V[i].round(3)}")
    
    # ── Step 2: Attention Scores ──
    scores = Q @ K.T
    print(f"\n{'='*60}")
    print("  STEP 2: Attention Scores = Q · K^T")
    print(f"{'='*60}")
    print(f"  Shape: {scores.shape} — each cell = dot product of Q_i and K_j\n")
    print_attention_map(scores, labels)
    
    # ── Step 3: Scale ──
    scaled = scores / np.sqrt(d_k)
    print(f"\n{'='*60}")
    print(f"  STEP 3: Scale by √d_k = √{d_k} = {np.sqrt(d_k):.1f}")
    print(f"{'='*60}")
    print(f"  This prevents softmax saturation when d_k is large.\n")
    print_attention_map(scaled, labels)
    
    # ── Step 4: Softmax ──
    attn_weights = softmax(scaled)
    print(f"\n{'='*60}")
    print("  STEP 4: Softmax (row-wise) → Attention Weights")
    print(f"{'='*60}")
    print(f"  Each row is now a probability distribution (sums to 1).\n")
    print_attention_map(attn_weights, labels)
    
    # ── Step 5: Output ──
    output = attn_weights @ V
    print(f"\n{'='*60}")
    print("  STEP 5: Output = Attention Weights · V")
    print(f"{'='*60}")
    print(f"  Shape: {output.shape} — each token is now context-enriched!\n")
    for i, label in enumerate(labels):
        components = " + ".join(
            f"{attn_weights[i,j]:.2f}×V_{labels[j]}" for j in range(seq_len)
        )
        print(f"  Output[{label:>5s}] = {components}")
        print(f"                  = {output[i].round(3)}\n")
    
    # ── BONUS: Causal Masking ──
    print(f"\n{'='*60}")
    print("  BONUS: CAUSAL (MASKED) SELF-ATTENTION (GPT-style)")
    print(f"{'='*60}")
    
    causal_mask = create_causal_mask(seq_len)
    print(f"\n  Causal Mask (True = can attend, False = blocked):")
    for i in range(seq_len):
        row = "  ".join(f"{'✓':>5s}" if causal_mask[i,j] else f"{'✗':>5s}" for j in range(seq_len))
        print(f"   {labels[i]:>5s}:  {row}")
    
    # Apply mask to scaled scores
    masked_scores = np.where(causal_mask, scaled, -np.inf)
    print(f"\n  Scaled scores AFTER mask (-inf blocks future tokens):")
    for i in range(seq_len):
        row = "  ".join(
            f"{masked_scores[i,j]:7.3f}" if causal_mask[i,j] else f"{'  -inf':>7s}"
            for j in range(seq_len)
        )
        print(f"   {labels[i]:>5s}:  {row}")
    
    causal_weights = softmax(masked_scores)
    print(f"\n  Attention weights with causal mask:")
    print_attention_map(causal_weights, labels)
    
    causal_output = causal_weights @ V
    print(f"\n  Causal output vectors:")
    for i, label in enumerate(labels):
        print(f"   {label:>5s}: {causal_output[i].round(3)}")
    
    # ── Key Observations ──
    print(f"\n{'='*60}")
    print("  KEY OBSERVATIONS")
    print(f"{'='*60}")
    print("""
  1. WITHOUT mask: "the" can see "cat" and "sat" (full bidirectional)
     → Used in BERT (encoder-only models) for understanding tasks

  2. WITH causal mask: "the" can only see itself
     → Used in GPT (decoder-only models) for generation
     → Ensures autoregressive property: predict next token from past only

  3. The attention weights tell us WHAT the model is paying attention to.
     This is why attention is so interpretable — you can visualize it!

  4. The output vectors are "context-enriched" — each token now carries
     information from the tokens it attended to, not just its own embedding.
""")
