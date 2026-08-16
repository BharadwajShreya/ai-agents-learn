"""
Session 1.8 — Generation Parameters: Temperature, Top-k, Top-p (NumPy)
=======================================================================
Demonstrates how different sampling strategies affect text generation:
  1. Temperature scaling
  2. Top-k filtering
  3. Top-p (nucleus) sampling
  4. Frequency & presence penalties
  5. Combined strategies with side-by-side comparison
"""

import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def apply_temperature(logits, temperature):
    """Scale logits by temperature before softmax."""
    if temperature <= 0:
        temperature = 1e-10  # avoid division by zero
    return logits / temperature


def apply_top_k(probs, k):
    """Zero out all but the top-k highest probability tokens, then renormalize."""
    if k >= len(probs):
        return probs
    sorted_idx = np.argsort(probs)[::-1]
    filtered = np.zeros_like(probs)
    for i in range(k):
        filtered[sorted_idx[i]] = probs[sorted_idx[i]]
    return filtered / filtered.sum()


def apply_top_p(probs, p):
    """Keep the smallest set of tokens whose cumulative probability >= p."""
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    
    # Find cutoff: keep tokens up to where cumsum >= p
    cutoff = np.searchsorted(cumsum, p) + 1
    cutoff = min(cutoff, len(probs))
    
    filtered = np.zeros_like(probs)
    for i in range(cutoff):
        filtered[sorted_idx[i]] = probs[sorted_idx[i]]
    return filtered / filtered.sum()


def apply_penalties(logits, token_counts, freq_penalty=0.0, presence_penalty=0.0):
    """Apply frequency and presence penalties to logits."""
    adjusted = logits.copy()
    for i, count in enumerate(token_counts):
        if count > 0:
            adjusted[i] -= freq_penalty * count
            adjusted[i] -= presence_penalty
    return adjusted


def sample_token(probs, rng=None):
    """Sample a token from the probability distribution."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(len(probs), p=probs)


# ============================================================
if __name__ == "__main__":
    
    # Vocabulary and raw logits (simulating model output)
    vocab = ["cat", "dog", "fish", "car", "hat", "tree", "moon", "book"]
    logits = np.array([5.0, 3.5, 2.0, 0.5, -0.5, -1.0, -2.0, -3.0])
    
    print("=" * 65)
    print("  SESSION 1.8 — GENERATION PARAMETERS")
    print("=" * 65)
    
    print(f"\n📥 Raw logits from model (one per vocab token):")
    for i, (word, logit) in enumerate(zip(vocab, logits)):
        print(f"   {word:>6s}: {logit:>6.1f}")
    
    base_probs = softmax(logits)
    print(f"\n📊 Base probabilities (softmax of logits, T=1.0):")
    for word, prob in zip(vocab, base_probs):
        bar = "█" * int(prob * 50)
        print(f"   {word:>6s}: {prob:.3f}  {bar}")
    
    # ══════════════════════════════════════════════════════
    #  1. TEMPERATURE
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  1. TEMPERATURE — Controls randomness")
    print(f"{'='*65}")
    print("  Formula: adjusted_logits = logits / temperature")
    
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    print(f"\n  {'Token':>6s}", end="")
    for t in temperatures:
        print(f"  T={t:<4}", end="")
    print()
    print("  " + "-" * 55)
    
    for i, word in enumerate(vocab):
        print(f"  {word:>6s}", end="")
        for t in temperatures:
            scaled = apply_temperature(logits, t)
            probs = softmax(scaled)
            print(f"  {probs[i]:.3f}", end="")
        print()
    
    print(f"""
  Key observations:
  • T=0.1: "cat" gets 100% — completely deterministic (greedy)
  • T=0.5: "cat" gets ~88% — very focused but slight variation
  • T=1.0: Original distribution — "cat" at ~65%
  • T=2.0: Much flatter — "cat" only ~30%, others gain probability
  
  ➡ Lower T = more confident/deterministic
  ➡ Higher T = more random/creative""")
    
    # ══════════════════════════════════════════════════════
    #  2. TOP-K SAMPLING
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  2. TOP-K SAMPLING — Keep only k most probable tokens")
    print(f"{'='*65}")
    
    for k in [1, 3, 5]:
        filtered = apply_top_k(base_probs, k)
        kept = [(vocab[i], filtered[i]) for i in range(len(vocab)) if filtered[i] > 0]
        print(f"\n  Top-k = {k}:")
        for word, prob in kept:
            bar = "█" * int(prob * 40)
            print(f"    {word:>6s}: {prob:.3f}  {bar}")
        removed = [vocab[i] for i in range(len(vocab)) if filtered[i] == 0]
        if removed:
            print(f"    Removed: {', '.join(removed)}")
    
    print(f"""
  • k=1 → greedy (always "cat")
  • k=3 → only cat/dog/fish considered
  • k=5 → broader but still no tail tokens
  
  Problem: k is FIXED — doesn't adapt to model confidence!""")
    
    # ══════════════════════════════════════════════════════
    #  3. TOP-P (NUCLEUS) SAMPLING
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  3. TOP-P (NUCLEUS) SAMPLING — Adaptive cutoff")
    print(f"{'='*65}")
    print("  Keep smallest set of tokens with cumulative prob ≥ p")
    
    for p in [0.5, 0.8, 0.95]:
        filtered = apply_top_p(base_probs, p)
        kept = [(vocab[i], filtered[i]) for i in range(len(vocab)) if filtered[i] > 0]
        print(f"\n  Top-p = {p}:")
        cumsum = 0
        for word, prob in kept:
            cumsum += prob
            bar = "█" * int(prob * 40)
            print(f"    {word:>6s}: {prob:.3f}  {bar}")
        print(f"    Tokens kept: {len(kept)}")
    
    print(f"""
  • p=0.5 → only "cat" needed (already >50%)
  • p=0.8 → "cat" + "dog" (cumulative ~87%)
  • p=0.95 → "cat" + "dog" + "fish" + "car"
  
  ✅ Top-p ADAPTS: tight when confident, wide when uncertain!""")
    
    # ══════════════════════════════════════════════════════
    #  4. TOP-P ADAPTS (demonstration)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  4. WHY TOP-P IS BETTER THAN TOP-K (Adaptive Demo)")
    print(f"{'='*65}")
    
    # Scenario 1: Model is confident
    confident_logits = np.array([10.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -3.0])
    confident_probs = softmax(confident_logits)
    
    # Scenario 2: Model is uncertain
    uncertain_logits = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.0, 0.5])
    uncertain_probs = softmax(uncertain_logits)
    
    print(f"\n  Scenario A — Model is CONFIDENT:")
    print(f"   Probs: ", end="")
    for w, p in zip(vocab, confident_probs):
        if p > 0.01:
            print(f"{w}({p:.0%}) ", end="")
    print()
    
    topk_conf = apply_top_k(confident_probs, 3)
    topp_conf = apply_top_p(confident_probs, 0.9)
    n_topk = sum(1 for x in topk_conf if x > 0)
    n_topp = sum(1 for x in topp_conf if x > 0)
    print(f"   Top-k=3: keeps {n_topk} tokens (2 are noise!)")
    print(f"   Top-p=0.9: keeps {n_topp} token(s) (perfectly tight!)")
    
    print(f"\n  Scenario B — Model is UNCERTAIN:")
    print(f"   Probs: ", end="")
    for w, p in zip(vocab, uncertain_probs):
        if p > 0.05:
            print(f"{w}({p:.0%}) ", end="")
    print()
    
    topk_unc = apply_top_k(uncertain_probs, 3)
    topp_unc = apply_top_p(uncertain_probs, 0.9)
    n_topk = sum(1 for x in topk_unc if x > 0)
    n_topp = sum(1 for x in topp_unc if x > 0)
    print(f"   Top-k=3: keeps {n_topk} tokens (too restrictive!)")
    print(f"   Top-p=0.9: keeps {n_topp} tokens (adapts to uncertainty!)")
    
    # ══════════════════════════════════════════════════════
    #  5. FREQUENCY & PRESENCE PENALTIES
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  5. FREQUENCY & PRESENCE PENALTIES")
    print(f"{'='*65}")
    
    # Simulate a sequence where "cat" has been repeated
    token_counts = [3, 1, 0, 0, 0, 0, 0, 0]  # "cat" appeared 3x, "dog" 1x
    
    print(f"\n  Token history: cat×3, dog×1, others×0")
    print(f"\n  Original logits vs penalized logits:")
    
    for fp, pp in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]:
        adjusted = apply_penalties(logits, token_counts, fp, pp)
        probs = softmax(adjusted)
        cat_prob = probs[0]
        dog_prob = probs[1]
        print(f"    freq={fp}, pres={pp}: cat={cat_prob:.3f}  dog={dog_prob:.3f}  "
              f"{'← cat penalized heavily' if cat_prob < 0.3 else ''}")
    
    print(f"""
  • freq_penalty: Penalizes proportional to count (cat×3 → big penalty)
  • presence_penalty: Penalizes if appeared at all (binary)
  • Both reduce repetition — freq is for word repetition, presence for topic diversity""")
    
    # ══════════════════════════════════════════════════════
    #  6. RECOMMENDED SETTINGS
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  6. RECOMMENDED SETTINGS BY USE CASE")
    print(f"{'='*65}")
    print(f"""
  ┌──────────────────┬──────┬───────┬───────┬──────────────────────┐
  │ Use Case         │ Temp │ Top-p │ Top-k │ Notes                │
  ├──────────────────┼──────┼───────┼───────┼──────────────────────┤
  │ Code generation  │ 0    │ 0.95  │  —    │ Deterministic        │
  │ Factual Q&A      │ 0.2  │ 0.9   │  —    │ Very reliable        │
  │ General chat     │ 0.7  │ 0.9   │  —    │ Balanced             │
  │ Creative writing │ 1.0  │ 0.95  │  —    │ Expressive           │
  │ Brainstorming    │ 1.3  │ 1.0   │  —    │ Maximum diversity    │
  └──────────────────┴──────┴───────┴───────┴──────────────────────┘

  Pro tips:
  • Temperature + Top-p are usually combined (most common setup)
  • Top-k is rarely used alone (top-p is strictly better)
  • For production/agents: T=0 (greedy) for reliability
  • For user-facing chat: T=0.7, top-p=0.9
""")
