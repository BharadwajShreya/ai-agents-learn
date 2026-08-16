"""
Session 1.5 — Transformer Block from Scratch (NumPy)
=====================================================
Builds on multi_head_attention.py — adds FFN, LayerNorm/RMSNorm,
and residual connections to create a complete Transformer block.

Covers:
  1. Feed-Forward Network (expand → activate → contract)
  2. LayerNorm and RMSNorm
  3. Residual connections (skip connections)
  4. Pre-Norm vs Post-Norm
  5. Full Transformer Block = MHA + FFN + Residuals + Norms
"""

import numpy as np


# ── Reuse from Session 1.4 ──
def softmax(x):
    """Row-wise softmax with numerical stability."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def single_head_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores)
    return weights @ V, weights


def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O, mask=None):
    """Multi-head attention: split → attend → concat → project."""
    head_outputs = []
    for i in range(len(W_Qs)):
        Q_i, K_i, V_i = X @ W_Qs[i], X @ W_Ks[i], X @ W_Vs[i]
        out_i, _ = single_head_attention(Q_i, K_i, V_i, mask)
        head_outputs.append(out_i)
    return np.concatenate(head_outputs, axis=-1) @ W_O


# ═══════════════════════════════════════════════════════
#  NEW COMPONENTS FOR SESSION 1.5
# ═══════════════════════════════════════════════════════

# ── 1. ACTIVATION FUNCTIONS ──

def relu(x):
    """ReLU: max(0, x) — original Transformer (2017)."""
    return np.maximum(0, x)


def gelu(x):
    """GELU: Gaussian Error Linear Unit — used by GPT-2, GPT-3, BERT.
    Smooth approximation of ReLU that allows small negative values through.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x):
    """Swish: x * sigmoid(x) — basis for SwiGLU."""
    return x * (1 / (1 + np.exp(-x)))


# ── 2. FEED-FORWARD NETWORK ──

def ffn(x, W1, b1, W2, b2, activation=gelu):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = activation(x · W1 + b1) · W2 + b2
    
    Args:
        x:    Input                (seq_len, d_model)
        W1:   Expand weights       (d_model, d_ff)      ← expand to 4× wider
        b1:   Expand bias          (d_ff,)
        W2:   Contract weights     (d_ff, d_model)       ← contract back
        b2:   Contract bias        (d_model,)
    
    Returns:
        output: (seq_len, d_model)  ← same shape as input
    """
    hidden = activation(x @ W1 + b1)   # (seq_len, d_ff)   — expand + activate
    output = hidden @ W2 + b2           # (seq_len, d_model) — contract
    return hidden, output


def ffn_swiglu(x, W1, W3, W2):
    """
    SwiGLU FFN — the modern standard (Llama, Mistral, Gemma).
    
    SwiGLU(x) = (Swish(x · W1) ⊙ (x · W3)) · W2
    
    Key difference: uses a GATING mechanism (element-wise multiply
    of two different linear projections) instead of a simple activation.
    """
    gate = swish(x @ W1)       # (seq_len, d_ff) — gating signal
    value = x @ W3             # (seq_len, d_ff) — value signal
    hidden = gate * value      # element-wise multiply (the "GLU" part)
    output = hidden @ W2       # (seq_len, d_model) — contract
    return hidden, output


# ── 3. LAYER NORMALIZATION ──

def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization — normalizes across the feature dimension (d_model).
    
    LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
    
    Applied INDEPENDENTLY to each token position.
    
    Args:
        x:     Input     (seq_len, d_model)
        gamma: Scale     (d_model,)  — learned
        beta:  Shift     (d_model,)  — learned
        eps:   Stability constant
    """
    mean = np.mean(x, axis=-1, keepdims=True)       # (seq_len, 1)
    var = np.var(x, axis=-1, keepdims=True)          # (seq_len, 1)
    x_norm = (x - mean) / np.sqrt(var + eps)         # normalize
    return gamma * x_norm + beta                      # scale and shift


def rms_norm(x, gamma, eps=1e-5):
    """
    RMSNorm — Root Mean Square Normalization.
    Used by Llama 2/3, Mistral, Gemma.
    
    RMSNorm(x) = x / √(mean(x²) + ε) · γ
    
    Simpler than LayerNorm:
      - No mean subtraction (no centering)
      - No β (no shift)
      - Faster to compute, same quality
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


# ── 4. TRANSFORMER BLOCK ──

def transformer_block(x, mha_params, ffn_params, norm_params, mask=None):
    """
    Complete Transformer Block (Pre-Norm style, as used by GPT-2+, Llama, etc.)
    
    Architecture:
        x → RMSNorm → MHA → + x (residual)
          → RMSNorm → FFN → + x (residual)
    
    Args:
        x:           Input embeddings    (seq_len, d_model)
        mha_params:  Dict with W_Qs, W_Ks, W_Vs, W_O
        ffn_params:  Dict with W1, b1, W2, b2
        norm_params: Dict with gamma1, gamma2
        mask:        Optional causal mask
    """
    # ── Sub-block 1: Multi-Head Attention ──
    x_norm1 = rms_norm(x, norm_params['gamma1'])           # Pre-norm
    attn_out = multi_head_attention(
        x_norm1,
        mha_params['W_Qs'], mha_params['W_Ks'],
        mha_params['W_Vs'], mha_params['W_O'],
        mask
    )
    x = x + attn_out                                        # Residual connection

    # ── Sub-block 2: Feed-Forward Network ──
    x_norm2 = rms_norm(x, norm_params['gamma2'])           # Pre-norm
    _, ffn_out = ffn(x_norm2,
                     ffn_params['W1'], ffn_params['b1'],
                     ffn_params['W2'], ffn_params['b2'])
    x = x + ffn_out                                         # Residual connection

    return x


# ============================================================
#  DEMO
# ============================================================
if __name__ == "__main__":
    # Same input from previous sessions
    X = np.array([
        [ 0.20,  1.42, -0.96,  1.94],   # "the"
        [ 1.07, -0.18, -0.41,  0.73],   # "cat"
        [ 1.09, -1.32,  0.24,  0.34],   # "sat"
    ])
    
    seq_len, d_model = X.shape   # 3, 4
    d_ff = d_model * 4           # 16 (expand 4×)
    h = 2                        # heads
    d_k = d_model // h           # 2
    labels = ['"the"', '"cat"', '"sat"']
    
    print("=" * 65)
    print("  SESSION 1.5 — TRANSFORMER BLOCK FROM SCRATCH")
    print("=" * 65)
    print(f"\n📥 Input X: shape ({seq_len}, {d_model})")
    print(f"⚙️  Config: d_model={d_model}, d_ff={d_ff} (4× expansion), h={h} heads")
    
    # ══════════════════════════════════════════════════════
    #  PART 1: ACTIVATION FUNCTIONS
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PART 1: Activation Functions Comparison")
    print(f"{'='*65}")
    
    test_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    print(f"\n  Input:  {test_vals}")
    print(f"  ReLU:   {relu(test_vals).round(3)}")
    print(f"  GELU:   {gelu(test_vals).round(3)}")
    print(f"  Swish:  {swish(test_vals).round(3)}")
    print("""
  Key differences:
  • ReLU:  Hard zero for negatives. Simple but loses information.
  • GELU:  Smooth. Small negative values get through. Used by GPT-2/3.
  • Swish: Similar to GELU. Basis for SwiGLU (Llama, Mistral).""")
    
    # ══════════════════════════════════════════════════════
    #  PART 2: FEED-FORWARD NETWORK
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PART 2: Feed-Forward Network (FFN)")
    print(f"{'='*65}")
    
    np.random.seed(55)
    W1 = np.random.randn(d_model, d_ff) * 0.2
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.2
    b2 = np.zeros(d_model)
    
    hidden, ffn_out = ffn(X, W1, b1, W2, b2, activation=gelu)
    
    print(f"\n  Input shape:   {X.shape}     = (seq_len, d_model)")
    print(f"  W1 shape:      {W1.shape}    = (d_model, d_ff) — EXPAND 4×")
    print(f"  Hidden shape:  {hidden.shape}  = (seq_len, d_ff) — expanded!")
    print(f"  W2 shape:      {W2.shape}   = (d_ff, d_model) — CONTRACT back")
    print(f"  Output shape:  {ffn_out.shape}     = (seq_len, d_model) — same as input ✓")
    
    print(f"\n  Parameter count in FFN:")
    ffn_params_count = d_model * d_ff + d_ff + d_ff * d_model + d_model
    print(f"    W1: {d_model}×{d_ff} = {d_model*d_ff}")
    print(f"    b1: {d_ff}")
    print(f"    W2: {d_ff}×{d_model} = {d_ff*d_model}")
    print(f"    b2: {d_model}")
    print(f"    Total: {ffn_params_count} params")
    print(f"\n  For Llama 3 8B (d_model=4096, d_ff=14336):")
    real_ffn = 2 * 4096 * 14336
    print(f"    FFN params per layer: ~{real_ffn:,} ({real_ffn/1e6:.0f}M)")
    print(f"    × 32 layers = ~{32*real_ffn/1e6:.0f}M params just for FFN!")
    
    for i, label in enumerate(labels):
        print(f"\n  {label} input:  {X[i].round(3)}")
        print(f"  {label} output: {ffn_out[i].round(3)}")
    
    # ══════════════════════════════════════════════════════
    #  PART 3: LAYER NORM vs RMS NORM
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PART 3: LayerNorm vs RMSNorm")
    print(f"{'='*65}")
    
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    
    ln_out = layer_norm(X, gamma, beta)
    rms_out = rms_norm(X, gamma)
    
    print(f"\n  Input (raw values have different scales per token):")
    for i, label in enumerate(labels):
        mean_val = np.mean(X[i])
        var_val = np.var(X[i])
        print(f"    {label}: {X[i].round(3)}  (mean={mean_val:.3f}, var={var_val:.3f})")
    
    print(f"\n  After LayerNorm (mean≈0, var≈1 per token):")
    for i, label in enumerate(labels):
        mean_val = np.mean(ln_out[i])
        var_val = np.var(ln_out[i])
        print(f"    {label}: {ln_out[i].round(3)}  (mean={mean_val:.6f}, var={var_val:.3f})")
    
    print(f"\n  After RMSNorm (no centering, just scale normalization):")
    for i, label in enumerate(labels):
        rms_val = np.sqrt(np.mean(rms_out[i]**2))
        print(f"    {label}: {rms_out[i].round(3)}  (RMS≈{rms_val:.3f})")
    
    print("""
  Key differences:
  • LayerNorm: Centers (subtracts mean) + scales. Has γ AND β.
  • RMSNorm:   Only scales (no centering). Has γ only. Faster.
  • Modern LLMs (Llama, Mistral): Use RMSNorm for efficiency.""")
    
    # ══════════════════════════════════════════════════════
    #  PART 4: RESIDUAL CONNECTIONS
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PART 4: Residual Connections")
    print(f"{'='*65}")
    
    print(f"\n  Without residual: output = Layer(x)")
    print(f"  With residual:    output = Layer(x) + x    ← just addition!")
    
    # Show how residual preserves information
    layer_output = ffn_out   # some transformed output
    residual_output = layer_output + X
    
    print(f"\n  Example for \"the\":")
    print(f"    Input x:       {X[0].round(3)}")
    print(f"    Layer(x):      {layer_output[0].round(3)}")
    print(f"    Layer(x) + x:  {residual_output[0].round(3)}")
    print(f"""
  Why residual connections are essential:
  1. Gradient flow: Even if Layer(x) has tiny gradients,
     the +x path has gradient = 1 (identity), so learning never stops.
  2. Incremental learning: Layer only learns what to ADD (the residual),
     not the entire transformation from scratch.
  3. Enables depth: Without residuals, 96-layer networks would be
     impossible to train (gradients vanish completely).""")
    
    # ══════════════════════════════════════════════════════
    #  PART 5: FULL TRANSFORMER BLOCK
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PART 5: Complete Transformer Block (Pre-Norm)")
    print(f"{'='*65}")
    print(f"  Architecture: x → RMSNorm → MHA → +x → RMSNorm → FFN → +x")
    
    # Initialize all parameters
    np.random.seed(42)
    mha_params = {
        'W_Qs': [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)],
        'W_Ks': [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)],
        'W_Vs': [np.random.randn(d_model, d_k) * 0.3 for _ in range(h)],
        'W_O':  np.random.randn(h * d_k, d_model) * 0.3,
    }
    ffn_p = {
        'W1': np.random.randn(d_model, d_ff) * 0.2,
        'b1': np.zeros(d_ff),
        'W2': np.random.randn(d_ff, d_model) * 0.2,
        'b2': np.zeros(d_model),
    }
    norm_p = {
        'gamma1': np.ones(d_model),
        'gamma2': np.ones(d_model),
    }
    
    # Run through the block step by step
    print(f"\n  Step-by-step trace:")
    
    x = X.copy()
    print(f"\n  0. Input:         shape {x.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {x[i].round(3)}")
    
    # Sub-block 1: MHA
    x_norm1 = rms_norm(x, norm_p['gamma1'])
    print(f"\n  1. After RMSNorm: shape {x_norm1.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {x_norm1[i].round(3)}")
    
    attn_out = multi_head_attention(
        x_norm1, mha_params['W_Qs'], mha_params['W_Ks'],
        mha_params['W_Vs'], mha_params['W_O']
    )
    print(f"\n  2. After MHA:     shape {attn_out.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {attn_out[i].round(3)}")
    
    x = x + attn_out   # residual
    print(f"\n  3. After +x (residual): shape {x.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {x[i].round(3)}")
    
    # Sub-block 2: FFN
    x_norm2 = rms_norm(x, norm_p['gamma2'])
    print(f"\n  4. After RMSNorm: shape {x_norm2.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {x_norm2[i].round(3)}")
    
    _, ffn_output = ffn(x_norm2, ffn_p['W1'], ffn_p['b1'], ffn_p['W2'], ffn_p['b2'])
    print(f"\n  5. After FFN:     shape {ffn_output.shape}")
    for i, label in enumerate(labels):
        print(f"     {label}: {ffn_output[i].round(3)}")
    
    x = x + ffn_output   # residual
    print(f"\n  6. After +x (residual) = FINAL OUTPUT:")
    for i, label in enumerate(labels):
        print(f"     {label}: {x[i].round(3)}")
    
    # Verify using the function
    block_out = transformer_block(X, mha_params, ffn_p, norm_p)
    match = np.allclose(x, block_out)
    print(f"\n  ✓ Manual trace matches transformer_block(): {'✅ Yes!' if match else '❌ No!'}")
    print(f"  ✓ Input shape == Output shape: {X.shape} == {block_out.shape} ✅")
    
    # ══════════════════════════════════════════════════════
    #  PARAMETER COUNT BREAKDOWN
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  PARAMETER COUNT — One Transformer Block")
    print(f"{'='*65}")
    
    mha_q_params = h * d_model * d_k
    mha_k_params = h * d_model * d_k
    mha_v_params = h * d_model * d_k
    mha_o_params = h * d_k * d_model
    mha_total = mha_q_params + mha_k_params + mha_v_params + mha_o_params
    
    ffn_total = d_model * d_ff + d_ff + d_ff * d_model + d_model
    norm_total = 2 * d_model  # gamma1 + gamma2
    
    block_total = mha_total + ffn_total + norm_total
    
    print(f"""
  Multi-Head Attention:
    W_Qs: {h} × ({d_model} × {d_k}) = {mha_q_params}
    W_Ks: {h} × ({d_model} × {d_k}) = {mha_k_params}
    W_Vs: {h} × ({d_model} × {d_k}) = {mha_v_params}
    W_O:  {h*d_k} × {d_model}       = {mha_o_params}
    MHA Total: {mha_total} params

  Feed-Forward Network:
    W1: {d_model} × {d_ff} = {d_model*d_ff}
    b1: {d_ff}
    W2: {d_ff} × {d_model} = {d_ff*d_model}
    b2: {d_model}
    FFN Total: {ffn_total} params

  Norms: 2 × {d_model} = {norm_total} params

  BLOCK TOTAL: {block_total} params
  FFN share: {ffn_total/block_total:.1%} of block  ← FFN dominates!

  For Llama 3 8B (d_model=4096, d_ff=14336, h=32, 32 layers):
    MHA per layer:  ~67M params
    FFN per layer:  ~117M params (FFN is ~63% of each layer!)
    Per layer:      ~184M params
    × 32 layers =   ~5.9B params (+ embeddings = ~8B total)
""")
    
    print(f"{'='*65}")
    print("  KEY TAKEAWAYS")
    print(f"{'='*65}")
    print("""
  1. A Transformer Block = MHA + FFN + Residuals + Norms
     - MHA: decides WHICH tokens to combine (relationships)
     - FFN: decides WHAT to do with combined info (knowledge/transforms)

  2. FFN uses expand → activate → contract pattern (d_model → 4×d_model → d_model)
     - FFN contains the MAJORITY of parameters (~63%)
     - Acts as a key-value memory storing learned knowledge

  3. Residual connections: output = Layer(x) + x
     - Prevents gradient vanishing in deep networks
     - Layer only learns the DELTA to add

  4. Pre-Norm (modern): Normalize BEFORE attention/FFN
     - More stable training than Post-Norm
     - RMSNorm is the current standard (simpler than LayerNorm)

  5. Input shape == Output shape: (seq_len, d_model)
     → This is why blocks can be STACKED arbitrarily!
""")
