"""
Session 1.4 — Multi-Head Attention from Scratch (NumPy)
=======================================================
Builds on attention.py — extends single-head self-attention to
multi-head attention with split → attend → concatenate → project.

Covers:
  1. Splitting d_model into h heads of size d_k = d_model / h
  2. Running independent attention on each head
  3. Concatenating head outputs
  4. Output projection with W_O
  5. Comparing single-head vs multi-head attention patterns
"""

import numpy as np


def softmax(x):
    """Row-wise softmax with numerical stability."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def single_head_attention(Q, K, V, mask=None):
    """Single-head scaled dot-product attention (from Session 1.3)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O, mask=None):
    """
    Multi-Head Attention.
    
    Args:
        X:     Input embeddings           (seq_len, d_model)
        W_Qs:  List of h query matrices   each (d_model, d_k)
        W_Ks:  List of h key matrices     each (d_model, d_k)
        W_Vs:  List of h value matrices   each (d_model, d_v)
        W_O:   Output projection matrix   (h * d_v, d_model)
        mask:  Optional causal mask       (seq_len, seq_len)
    
    Returns:
        output:      Final output          (seq_len, d_model)
        all_weights: Attention weights per head  list of (seq_len, seq_len)
    """
    h = len(W_Qs)
    head_outputs = []
    all_weights = []
    
    for i in range(h):
        # Each head computes its own Q, K, V in a smaller subspace
        Q_i = X @ W_Qs[i]    # (seq_len, d_k)
        K_i = X @ W_Ks[i]    # (seq_len, d_k)
        V_i = X @ W_Vs[i]    # (seq_len, d_v)
        
        # Standard scaled dot-product attention per head
        head_out, head_weights = single_head_attention(Q_i, K_i, V_i, mask)
        head_outputs.append(head_out)     # (seq_len, d_v)
        all_weights.append(head_weights)  # (seq_len, seq_len)
    
    # Concatenate all head outputs along the last dimension
    # h heads × d_v each = d_model
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, h * d_v)
    
    # Output projection: mix information across heads
    output = concat @ W_O  # (seq_len, d_model)
    
    return output, all_weights


def print_attention_map(weights, labels, head_name=""):
    """Pretty-print attention weights."""
    n = len(labels)
    if head_name:
        print(f"\n  {head_name}:")
    header = "         " + "  ".join(f"{l:>7s}" for l in labels)
    print(header)
    for i in range(n):
        row = "  ".join(f"{weights[i, j]:7.3f}" for j in range(n))
        print(f'{labels[i]:>7s}  {row}')


# ============================================================
#  DEMO: Multi-Head Attention on "the cat sat"
# ============================================================
if __name__ == "__main__":
    # ── Input: Same final embeddings from Session 1.2 ──
    X = np.array([
        [ 0.20,  1.42, -0.96,  1.94],   # "the" (pos 0)
        [ 1.07, -0.18, -0.41,  0.73],   # "cat" (pos 1)
        [ 1.09, -1.32,  0.24,  0.34],   # "sat" (pos 2)
    ])
    
    seq_len, d_model = X.shape   # 3, 4
    labels = ['"the"', '"cat"', '"sat"']
    
    print("=" * 65)
    print("  SESSION 1.4 — MULTI-HEAD ATTENTION FROM SCRATCH")
    print("=" * 65)
    
    print(f"\n📥 Input X shape: ({seq_len}, {d_model}) = (seq_len, d_model)")
    
    # ── Configuration ──
    h = 2           # number of heads
    d_k = d_model // h   # 4 // 2 = 2 per head
    
    print(f"\n⚙️  Config: h={h} heads, d_k = d_model/h = {d_model}/{h} = {d_k}")
    print(f"   Each head works in a {d_k}-dimensional subspace")
    print(f"   Total params in attention: same as single head with d_k={d_model}")
    
    # ── Initialize weight matrices for each head ──
    np.random.seed(42)
    W_Qs = [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)]
    W_Ks = [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)]
    W_Vs = [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)]
    W_O  = np.random.randn(h * d_k, d_model) * 0.3   # (4, 4)
    
    # ══════════════════════════════════════════════════════
    #  STEP 1: Show what each head does independently
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  STEP 1: Each head computes attention in its own subspace")
    print(f"{'='*65}")
    
    head_outputs = []
    head_weights = []
    
    for i in range(h):
        Q_i = X @ W_Qs[i]
        K_i = X @ W_Ks[i]
        V_i = X @ W_Vs[i]
        
        print(f"\n  ── Head {i+1} ──")
        print(f"  Q_{i+1} = X · W_Q_{i+1}  →  shape {Q_i.shape}  (projects to {d_k}D subspace)")
        for j, label in enumerate(labels):
            print(f"    Q_{i+1}[{label}] = {Q_i[j].round(3)}")
        
        out_i, w_i = single_head_attention(Q_i, K_i, V_i)
        head_outputs.append(out_i)
        head_weights.append(w_i)
    
    # ══════════════════════════════════════════════════════
    #  STEP 2: Compare attention patterns across heads
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  STEP 2: Attention patterns — DIFFERENT per head! 👀")
    print(f"{'='*65}")
    print("  This is the KEY insight: each head learns different relationships.")
    
    for i in range(h):
        print_attention_map(head_weights[i], labels, f"Head {i+1} Attention Weights")
    
    # Highlight differences
    print(f"\n  📊 Comparison — who does each token attend to most?")
    for j, label in enumerate(labels):
        for i in range(h):
            max_idx = np.argmax(head_weights[i][j])
            max_val = head_weights[i][j][max_idx]
            print(f"    {label} in Head {i+1}: focuses on {labels[max_idx]} ({max_val:.1%})")
    
    # ══════════════════════════════════════════════════════
    #  STEP 3: Concatenate head outputs
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  STEP 3: Concatenate all head outputs")
    print(f"{'='*65}")
    
    concat = np.concatenate(head_outputs, axis=-1)
    print(f"\n  Head 1 output shape: {head_outputs[0].shape}")
    print(f"  Head 2 output shape: {head_outputs[1].shape}")
    print(f"  Concatenated shape:  {concat.shape}  ← {h} × {d_k} = {h*d_k} = d_model ✓")
    
    for j, label in enumerate(labels):
        h1 = head_outputs[0][j].round(3)
        h2 = head_outputs[1][j].round(3)
        print(f"\n  {label}: [{h1}] ⊕ [{h2}]")
        print(f"        = {concat[j].round(3)}")
    
    # ══════════════════════════════════════════════════════
    #  STEP 4: Output projection (W_O)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  STEP 4: Output Projection (W_O) — mixes information across heads")
    print(f"{'='*65}")
    
    final_output = concat @ W_O
    print(f"\n  Concat shape: {concat.shape}")
    print(f"  W_O shape:    {W_O.shape}")
    print(f"  Final output: {final_output.shape}  ← back to (seq_len, d_model) ✓")
    
    for j, label in enumerate(labels):
        print(f"  {label}: {final_output[j].round(3)}")
    
    # ══════════════════════════════════════════════════════
    #  STEP 5: Verify using the multi_head_attention function
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  STEP 5: Verify — using multi_head_attention() function")
    print(f"{'='*65}")
    
    mha_output, mha_weights = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
    match = np.allclose(final_output, mha_output)
    print(f"\n  Manual computation matches function: {'✅ Yes!' if match else '❌ No!'}")
    
    # ══════════════════════════════════════════════════════
    #  PARAMETER COUNT COMPARISON
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PARAMETER COUNT — Single-Head vs Multi-Head")
    print(f"{'='*65}")
    
    # Single head: W_Q + W_K + W_V = 3 × (d_model × d_model)
    single_params = 3 * d_model * d_model
    # Multi-head: h × (W_Q_i + W_K_i + W_V_i) + W_O = h × 3 × (d_model × d_k) + (d_model × d_model)
    multi_params = h * 3 * (d_model * d_k) + (h * d_k * d_model)
    
    print(f"""
  Single-Head (d_k = d_model = {d_model}):
    W_Q + W_K + W_V = 3 × ({d_model} × {d_model}) = {single_params} params
    Total: {single_params} params, 1 attention pattern

  Multi-Head ({h} heads, d_k = {d_k}):
    {h} × (W_Q_i + W_K_i + W_V_i) = {h} × 3 × ({d_model} × {d_k}) = {h * 3 * d_model * d_k} params
    + W_O = {h*d_k} × {d_model} = {h*d_k*d_model} params
    Total: {multi_params} params, {h} attention patterns

  Same total parameters, but {h}× richer representation! 🎯
""")
    
    # ══════════════════════════════════════════════════════
    #  KEY TAKEAWAYS
    # ══════════════════════════════════════════════════════
    print(f"{'='*65}")
    print("  KEY TAKEAWAYS")
    print(f"{'='*65}")
    print("""
  1. Multi-head attention runs h parallel attention operations,
     each in a smaller d_k-dimensional subspace (d_k = d_model/h).

  2. Different heads learn DIFFERENT attention patterns:
     - Some heads learn syntactic relationships (subject → verb)
     - Others learn coreference (pronoun → antecedent)
     - Others learn positional patterns (attend to neighbors)

  3. The output projection W_O mixes information ACROSS heads,
     allowing the model to combine multiple perspectives.

  4. Total parameter count ≈ same as single-head attention,
     but multi-head gives h× richer representations.

  5. This is the standard in ALL modern LLMs:
     GPT-2 uses 12 heads, GPT-3 uses 96 heads, Llama 3 uses 32-64 heads.
""")
