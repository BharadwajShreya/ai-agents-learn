# Module 2: Reasoning Models, Multimodal AI & Test-Time Compute

> **Purpose:** This module covers the paradigm shift from "instant answer" models to "think before you answer" models, multimodal AI architectures, and the 2026 model landscape.

---

## Session 2.1: System 1 vs System 2 Thinking + Reasoning Models

### 1. THE PROBLEM — Why Do We Need Reasoning Models?

Standard LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash) are **System 1 thinkers** — they generate each token in a **single forward pass** through the network, using **fixed compute** regardless of problem difficulty.

This means they spend the **same computation** on "What's 2+2?" and "Prove the Riemann hypothesis."

**Where this breaks:**

```
Problem: A bat and a ball cost $1.10 in total.
         The bat costs $1.00 more than the ball.
         How much does the ball cost?

Standard LLM answer: $0.10  ❌ (instinctive but WRONG)
Correct answer:      $0.05  ✅ (ball=0.05, bat=1.05, total=1.10)
```

Standard LLMs fail because they pattern-match to the "obvious" answer without a mechanism to self-verify.

---

### 2. THE NAIVE ALTERNATIVE — "Why Not Just Add 'Think Step by Step'?"

Adding "think step by step" to a standard LLM prompt **does NOT create a reasoning model**. Here's why:

| Aspect | CoT Prompting on Standard LLM | Actual Reasoning Model |
|--------|-------------------------------|----------------------|
| What happens internally | Each token is still 1 forward pass — no change in computation | Generates many hidden thinking tokens — more forward passes |
| Training | Not trained to reason step-by-step | Trained with Process Reward Models (PRMs) that reward each step |
| Self-correction | Cannot verify its own intermediate steps | Can detect and fix errors mid-reasoning |
| Output format | Steps are visible but may be **performance of reasoning** (theater) | Steps are hidden (hidden CoT) — genuine internal computation |
| Compute scaling | Fixed compute regardless of difficulty | Variable compute — harder problems get more thinking tokens |

**Key insight:** CoT prompting changes the OUTPUT FORMAT, not the COMPUTATION. The model still generates each token with a single forward pass. A reasoning model generates hundreds of hidden tokens, each one an additional forward pass, effectively giving the model a "scratchpad" for multi-step computation.

---

### 3. THE SOLUTION — System 1 vs System 2 Framework

Daniel Kahneman's framework from "Thinking, Fast and Slow" maps directly to LLMs:

```
┌────────────────────────────────┬────────────────────────────────────┐
│       SYSTEM 1 (Fast)          │        SYSTEM 2 (Slow)             │
├────────────────────────────────┼────────────────────────────────────┤
│ Automatic, instinctive         │ Deliberate, analytical             │
│ Low effort                     │ High effort                        │
│ "What's 2 + 2?"  → "4!"       │ "What's 17 × 24?" → *calculates*  │
│ Answers in milliseconds        │ Takes seconds to minutes           │
│ Pattern matching               │ Step-by-step reasoning             │
│ Often right, sometimes wrong   │ Slower but more reliable           │
│ Can't explain its reasoning    │ Can show its work                  │
├────────────────────────────────┼────────────────────────────────────┤
│  STANDARD LLMs                 │  REASONING MODELS                  │
│  GPT-4o, Claude 3.5 Sonnet,   │  o1, o3, o4-mini,                  │
│  Gemini 2.0 Flash              │  Claude with extended thinking,    │
│                                │  Gemini 2.5 Pro                    │
│  1 forward pass per token      │  Many internal "thinking" tokens   │
│  Fixed compute per problem     │  Scales compute to difficulty      │
└────────────────────────────────┴────────────────────────────────────┘
```

---

### 4. HOW REASONING MODELS WORK — Three Key Mechanisms

#### Mechanism 1: Hidden Chain-of-Thought (Hidden CoT)

The model generates **thinking tokens** that are **invisible to the user**. You only see the final answer, but internally the model may have generated hundreds or thousands of reasoning tokens.

```
What YOU see:                    What ACTUALLY happened:
┌──────────────────┐            ┌──────────────────────────────────┐
│ Q: [complex math] │            │ <thinking>                       │
│ A: 42             │            │   First, let me identify...      │
│                    │            │   If I try approach A...         │
│ (looks instant!)   │            │   Wait, that doesn't work...    │
│                    │            │   Let me try approach B...       │
│                    │            │   Yes! This gives me...         │
│                    │            │   Let me verify: ...            │
│                    │            │   Confirmed! The answer is 42.  │
│                    │            │ </thinking>                      │
│                    │            │ 42                               │
└──────────────────┘            └──────────────────────────────────┘
```

Each thinking token is a forward pass through the model, so hidden CoT effectively gives the model **many more computation steps** for hard problems. The model uses its own output as a scratchpad.

> **Note:** Some models like Claude show a summary of their thinking. OpenAI's o-series models hide it completely. The raw thinking tokens are never exposed via API.

#### Mechanism 2: Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)

This is the **training secret** behind reasoning models.

```
PROBLEM: "Solve 3x + 7 = 22"

=== ORM (Outcome Reward Model) ===
Only checks: Did you get the right final answer?

  Attempt 1: "x = 5"   →  ✅ Correct answer  →  Reward: +1
  Attempt 2: "x = 7"   →  ❌ Wrong answer    →  Reward: 0

  Problem: The model could get the right answer via WRONG reasoning!
  "3 + 7 = 10, 22 - 10 = 12, 12/3 = ... uh... 5?"  ← lucky guess rewarded!


=== PRM (Process Reward Model) ===
Checks EACH STEP of reasoning individually:

  Step 1: "3x + 7 = 22"         →  ✅ Correct setup      →  +1
  Step 2: "3x = 22 - 7 = 15"    →  ✅ Correct subtraction →  +1
  Step 3: "x = 15 / 3 = 5"      →  ✅ Correct division    →  +1

  Total reward: +3 (every step was valid!)

  Wrong path caught immediately:
  Step 1: "3x + 7 = 22"         →  ✅  →  +1
  Step 2: "3x = 22 + 7 = 29"    →  ❌  →  -1  ← ERROR CAUGHT HERE!
```

| Aspect | ORM | PRM |
|--------|-----|-----|
| What it rewards | Final answer only | Each intermediate step |
| Can reward lucky guesses? | Yes ❌ | No ✅ |
| Training signal density | Sparse (1 signal per problem) | Dense (1 signal per step) |
| Teaches self-correction? | No | Yes — model learns which steps lead to errors |
| Used in | Standard RLHF | Reasoning models (o1, o3, etc.) |

**Why PRMs matter:** They teach the model that **the process matters, not just the destination**. The model learns to recognize valid vs invalid reasoning steps, which enables reliable self-correction during hidden CoT.

#### Mechanism 3: Test-Time Compute Scaling

The revolutionary idea: instead of making models bigger at training time, give them **more time to think at inference time**.

```
                    COMPUTE ALLOCATION

    Traditional Scaling               Test-Time Scaling
    (Bigger Model)                    (More Thinking)

    ┌──────────────┐                  ┌──────────────┐
    │              │                  │   Thinking    │
    │   BIGGER     │                  │   Tokens      │
    │   WEIGHTS    │                  │   (variable)  │
    │              │                  │              ↕│ ← scales with
    │   (fixed     │                  │   ┌────────┐ │    difficulty
    │    cost per  │                  │   │ Small  │ │
    │    query)    │                  │   │ Model  │ │
    │              │                  │   └────────┘ │
    └──────────────┘                  └──────────────┘

    Cost: $$$$ always                  Cost: $ easy, $$$$ hard
```

**Practical cost implications:**

| Task | Standard LLM | Reasoning Model | Winner |
|------|-------------|-----------------|--------|
| "Summarize this article" | 0.3s, ~$0.001 | 3s, ~$0.05 | ⚡ Standard — no reasoning needed |
| "Write a marketing email" | 1s, ~$0.003 | 5s, ~$0.08 | ⚡ Standard — creative, not logical |
| "Solve competition math" | 0.5s, WRONG | 30s, ~$0.50, RIGHT | 🧠 Reasoning — multi-step logic |
| "Debug complex code" | 1s, partial fix | 15s, ~$0.30, full fix | 🧠 Reasoning — systematic analysis |
| "Design a system architecture" | 2s, shallow | 45s, ~$1.00, thorough | 🧠 Reasoning — planning & trade-offs |

---

### 5. REASONING MODELS vs ReAct AGENTS — Key Distinction

These are **fundamentally different concepts** that students often confuse:

```
┌─────────────────────────────────────────┬────────────────────────────────┐
│  REASONING MODEL (Hidden CoT)           │  ReAct AGENT PATTERN           │
├─────────────────────────────────────────┼────────────────────────────────┤
│  INTERNAL thinking only                 │  EXTERNAL actions + tools      │
│  No access to outside world             │  Calls APIs, searches, DBs     │
│  Single model, single turn              │  Loop: Think → Act → Observe   │
│  "Think hard, then answer"              │  "Think, do something,         │
│                                         │   see result, repeat"          │
│                                         │                                │
│  Example:                               │  Example:                      │
│  <thinking>                             │  Thought: I need weather data  │
│    Let me solve step by step...         │  Action: call_weather("NYC")   │
│    17 × 24 = 17 × 20 + 17 × 4...       │  Observation: 72°F, sunny      │
│  </thinking>                            │  Thought: Now I can answer     │
│  Answer: 408                            │  Answer: It's 72°F in NYC      │
│                                         │                                │
│  No external tools involved!            │  Uses external tools!          │
│  Covered: Module 2 (this module)        │  Covered: Module 7             │
└─────────────────────────────────────────┴────────────────────────────────┘
```

> **Common Misconception ⚠️:** Reasoning models and ReAct agents can be **combined**. You can use a reasoning model (like o3) as the LLM backbone inside a ReAct agent. The reasoning model handles the "Thought" step with deeper internal computation, while the ReAct loop handles external tool interactions. They operate at different levels.

---

### 6. COMPARISON TABLE — Standard LLM vs Reasoning Model

| Dimension | Standard LLM | Reasoning Model |
|-----------|-------------|-----------------|
| **Examples** | GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash | o1, o3, o4-mini, Claude with extended thinking, Gemini 2.5 Pro |
| **Compute per token** | 1 forward pass (fixed) | 1 forward pass per token, but generates many hidden tokens |
| **Total compute per query** | Fixed | Variable — scales with problem difficulty |
| **Training reward** | ORM (outcome only) | PRM (process — each step rewarded) |
| **Self-correction** | No mechanism | Yes — trained to detect/fix errors mid-chain |
| **Visible output** | All tokens visible | Hidden thinking tokens + visible final answer |
| **Latency** | Fast (sub-second for short answers) | Slow (seconds to minutes for hard problems) |
| **Cost** | Low, predictable | Higher, variable |
| **Best for** | Creative writing, summarization, translation, simple Q&A | Math, logic, code debugging, planning, complex analysis |
| **Worst for** | Multi-step math, logic puzzles, systematic debugging | Simple tasks (overkill — slower and more expensive) |

---

### 7. INTERVIEW ANGLES

> **Q: "What is a reasoning model and how does it differ from a standard LLM?"**
>
> **Model Answer:** "A reasoning model generates hidden chain-of-thought tokens before producing the final answer — each token is a forward pass, so the model gets many more computation steps for hard problems. These models are trained with Process Reward Models (PRMs) that reward each reasoning step, not just the final outcome, enabling self-correction mid-chain. Standard LLMs generate each token with a single forward pass using fixed compute regardless of difficulty. Adding 'think step by step' to a standard LLM only changes the output format — it doesn't add computation or self-verification."

> **Q: "When would you NOT use a reasoning model?"**
>
> **Model Answer:** "For tasks that don't require multi-step logical reasoning — creative writing, summarization, translation, simple Q&A. Reasoning models are slower and more expensive because they generate many hidden tokens. For a chatbot handling customer FAQ, using o3 instead of GPT-4o would increase latency from 0.3s to 5s+ and costs by 10-50x with no quality improvement."

> **Q: "What's the difference between a PRM and an ORM?"**
>
> **Model Answer:** "An Outcome Reward Model only evaluates the final answer — did you get it right or wrong. A Process Reward Model evaluates every intermediate reasoning step. PRMs provide denser training signal and prevent the model from being rewarded for getting the right answer through wrong reasoning (lucky guesses). This is critical for teaching reliable multi-step reasoning and self-correction."

---

### 8. CODE REFERENCE

- See `module_01_llm_internals/notes.md` for prerequisite concepts on forward passes and transformer architecture

---
---

## Session 2.2: Test-Time Compute, Deep Research & Prompting Reasoning Models

### 1. THE PROBLEM — Why Can't We Just "Always Think More"?

If more thinking tokens = better answers, why not always generate 10,000 thinking tokens?

Because of **diminishing returns + cost explosion**:

```
                    ACCURACY vs THINKING TOKENS

  Accuracy
  100% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ceiling
   95% │                          ●━━━━━━━━━●━━━━━━━━━●
       │                     ●
   85% │                ●
       │           ●
   70% │      ●                        ↑ Diminishing returns!
       │  ●                            More tokens ≠ proportionally
   50% │●                              more accuracy
       │
       └──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──→
          10  50  100 200 500 1K  2K  5K  10K
                    Thinking Tokens Generated

  Cost:  $   $   $$  $$  $$$ $$$ $$$$ $$$$ $$$$$$
```

**Key finding:** There's a **compute-optimal frontier** — for each problem difficulty, there's a sweet spot of thinking tokens beyond which you're just burning money.

---

### 2. TWO STRATEGIES FOR TEST-TIME COMPUTE

#### Strategy 1: Sequential Refinement (Think longer on ONE chain)

```
Start → Think → Think more → Think even more → Answer

"Let me re-examine step 3... actually that's wrong..."
"Wait, I should consider edge case X..."
"Let me verify by working backwards..."
```

- ✅ Good for: Deep, complex single problems
- ❌ Risk: Can get stuck in wrong reasoning path

#### Strategy 2: Parallel Sampling (Try MANY chains, pick the best)

```
Start ─┬─→ Chain A → Answer A ─┐
       ├─→ Chain B → Answer B ──┤─→ Pick best (majority vote
       ├─→ Chain C → Answer C ──┤    or verifier scoring)
       └─→ Chain D → Answer D ─┘
```

- ✅ Good for: Problems with verifiable answers (math, code)
- ❌ Risk: Expensive (N× the cost), wasteful if chains agree early

**In practice, reasoning models use BOTH:**

```
Problem arrives
      │
      ▼
┌─────────────┐     "Is this easy or hard?"
│  Difficulty  │     (learned during training —
│  Assessment  │      not a separate classifier)
└──────┬──────┘
       │
   ┌───┴───┐
 Easy     Hard
   │       │
   ▼       ▼
 Short    Long sequential chain
 chain    + parallel sampling
 ~50      + self-verification
 tokens   ~500-5000 tokens
```

---

### 3. THE DEEP RESEARCH PATTERN

Deep Research is one of the most impressive applications of reasoning models — **multi-step autonomous research**.

```
  User: "Write a comprehensive analysis of X"
                    │
         ┌──────────┴───────────┐
         │  REASONING MODEL     │
         │  (the "brain")       │
         └──────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
  PLAN            SEARCH          SYNTHESIZE

  Step 1: Break    Step 2: Search   Step 3: Read &
  question into    web for each     extract key
  sub-questions    sub-question     findings
         │              │                │
         └──────────────┴────────────────┘
                        │
                        ▼
                 Step 4: Gap Analysis
                 "I found gaps in my research"
                 → search more → repeat 2-5 times
                        │
                        ▼
                 Step 5: Compile final report
                 with citations
                 
  Total time: 5-30 minutes
  Total searches: 10-100+
  Total tokens: 50,000-500,000+
```

#### Why Deep Research NEEDS Reasoning Models

| Component | Role | Why reasoning model needed |
|-----------|------|--------------------------|
| **Planning** | Break question into sub-questions | Requires multi-step decomposition |
| **Search orchestration** | Decide what to search, when to stop | Requires judgment about information sufficiency |
| **Source evaluation** | Assess reliability, detect contradictions | Requires critical reasoning |
| **Gap analysis** | "What's missing from my research?" | Requires self-reflection |
| **Synthesis** | Combine findings into coherent report | Requires organizing complex information |

A standard LLM would just write from its training data and potentially hallucinate citations. A reasoning model with tools actually searches, reads, evaluates, and iterates.

---

### 4. HOW TO PROMPT REASONING MODELS DIFFERENTLY

> **Critical Rule:** Tell reasoning models WHAT you want, not HOW to think.

#### Comparison Table

| Technique | Standard LLM | Reasoning Model |
|-----------|-------------|-----------------|
| "Think step by step" | ✅ Helps format output | ❌ **Harmful** — interferes with trained reasoning process |
| Few-shot examples | ✅ 3-5 examples help | ⚠️ Can hurt — model matches format instead of reasoning freely |
| Detailed instructions | ✅ Specify format, style | ✅ Keep clear but DON'T over-constrain reasoning |
| System prompt | ✅ Key for persona/style | ⚠️ Some models (o1) ignored these; o3/o4 improved |
| Temperature | ✅ Controls randomness | ❌ Use `reasoning_effort` instead |
| Task decomposition | ✅ Break into sub-tasks | ✅ Give the WHOLE task — let the model decompose |

#### Why "Think Step by Step" Hurts Reasoning Models

The model was trained with PRMs to follow its own optimized reasoning process. When you prescribe steps like "first identify the problem type, then list formulas...", you:
1. Force the model into YOUR reasoning structure instead of its (better) trained one
2. Cause it to generate redundant thinking (your steps + its internal steps)
3. Make it try to match your format rather than reason freely

#### The `reasoning_effort` Parameter

| Effort Level | Thinking Tokens | Latency | Best For |
|-------------|----------------|---------|----------|
| **LOW** | ~50-200 | 1-3 sec | Simple classification, routing decisions |
| **MEDIUM** (default) | ~200-2000 | 3-15 sec | Code generation, analysis tasks |
| **HIGH** | ~2000-50000+ | 15-120 sec | Competition math, complex debugging |

**Code example:**
```python
# Standard LLM call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Solve this..."}],
    temperature=0.7,          # ← controls randomness
)

# Reasoning model call
response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": "Solve this..."}],
    reasoning_effort="high",  # ← controls how MUCH to think
    # No temperature needed — reasoning handles exploration internally
)
```

#### Prompting Examples

**❌ WRONG — Prompting o3 like it's GPT-4o:**
```
System: You are a math tutor. Think step by step. First identify
        the problem type, then list formulas, then solve, then verify.
User: Solve ∫(x² · eˣ) dx
```

**✅ RIGHT — Prompting o3 properly:**
```
User: Solve ∫(x² · eˣ) dx. Show the final solution with verification.
```

Short, direct, focused on WHAT (answer + verification), not HOW to think.

---

### 5. DECISION FRAMEWORK — When to Use What

```
                 WHICH MODEL SHOULD I USE?

                    ┌─────────────┐
                    │ Does task    │
                    │ need multi-  │
                    │ step logic?  │
                    └──────┬──────┘
                     ╱          ╲
                  YES             NO
                   │               │
                   ▼               ▼
          ┌────────────┐   ┌─────────────┐
          │ Is latency  │   │ Standard LLM │
          │ critical    │   │ (GPT-4o,     │
          │ (<2 sec)?   │   │  Claude 3.5,  │
          └─────┬──────┘   │  Gemini Flash)│
           ╱         ╲     └─────────────┘
        YES            NO
         │              │
         ▼              ▼
  ┌────────────┐  ┌──────────────┐
  │ Standard   │  │ Is budget    │
  │ LLM + CoT  │  │ unlimited?   │
  │ prompting  │  └──────┬───────┘
  │ (best      │    ╱         ╲
  │  effort)   │  YES           NO
  └────────────┘   │            │
                   ▼            ▼
            ┌──────────┐  ┌──────────────┐
            │ Full      │  │ Reasoning    │
            │ reasoning │  │ model with   │
            │ (high     │  │ effort=medium│
            │  effort)  │  │ (balanced)   │
            └──────────┘  └──────────────┘
```

#### Quick Reference

| Task | Recommended Model | Why |
|------|------------------|-----|
| Customer support chatbot | Standard LLM | Speed matters, pattern-based answers |
| Summarize a document | Standard LLM | No multi-step logic needed |
| Write marketing copy | Standard LLM | Creative task, not logical |
| Solve coding challenge | Reasoning (medium) | Multi-step problem solving |
| Debug complex system | Reasoning (high) | Systematic analysis needed |
| Competition math | Reasoning (high) | Deep multi-step reasoning |
| Classify email sentiment | Standard LLM / SLM | Trivially simple |
| Research report with sources | Reasoning + Deep Research | Planning + search + synthesis |
| Real-time autocomplete | Standard LLM / SLM | Latency is everything |

---

### 6. PRACTICAL MODEL ROUTING ARCHITECTURE

For systems handling mixed-complexity queries (e.g., 10K customer support queries/day):

```
  All Queries (10,000/day)
         │
         ▼
  ┌──────────────────┐
  │  LIGHTWEIGHT      │   ← Must be CHEAP and FAST
  │  ROUTER/          │      (SLM or fine-tuned classifier)
  │  CLASSIFIER       │      NOT a reasoning model!
  └────────┬─────────┘
           │
    ┌──────┼──────┐
    │      │      │
    ▼      ▼      ▼
  Simple  Medium  Complex
  (~80%)  (~15%)  (~5%)
    │      │      │
    ▼      ▼      ▼
  SLM/   Standard Reasoning
  Fast   LLM     Model
  LLM            (high effort)

  Cost: 8000×$0.001 + 1500×$0.01 + 500×$0.30
      = $8 + $15 + $150 = $173/day

  VS all reasoning: 10000×$0.30 = $3,000/day  (17× more expensive!)
```

> **Interview Tip 💡:** "In production, model routing is crucial for cost optimization. Use a three-tier architecture: SLM for trivial queries, standard LLM for moderate, reasoning model for complex. The router itself must be cheap — use a fine-tuned small classifier, not a reasoning model. This can reduce costs by 10-20× compared to routing everything through a reasoning model."

---

### 7. INTERVIEW ANGLES

> **Q: "How does test-time compute scaling work?"**
>
> **Model Answer:** "Instead of making models bigger at training time, test-time compute scales by generating more hidden thinking tokens at inference. The model uses two strategies: sequential refinement (thinking deeper on one chain with self-correction) and parallel sampling (generating multiple reasoning chains and picking the best via majority vote or verifier scoring). The key insight is the compute-optimal frontier — for each problem difficulty, there's a sweet spot beyond which additional thinking gives diminishing returns."

> **Q: "How would you design model selection for a production system with mixed query complexity?"**
>
> **Model Answer:** "I'd use a three-tier architecture with a lightweight router. A cheap classifier (SLM or fine-tuned small model) categorizes queries by complexity. ~80% are simple and go to a fast standard LLM, ~15% medium-complexity go to a standard LLM with detailed prompting, and ~5% complex queries go to a reasoning model. This can reduce costs by 10-20× vs sending everything to a reasoning model, while maintaining quality on the queries that actually need deep reasoning."

> **Q: "Why should you prompt reasoning models differently?"**
>
> **Model Answer:** "Reasoning models are trained with Process Reward Models to follow their own optimized reasoning process. Over-constraining with instructions like 'think step by step' or prescribing specific reasoning steps can interfere with this trained process — the model may try to match your format instead of reasoning freely, or generate redundant thinking. The best practice is to specify WHAT you want (the answer, verification, format) not HOW to think. Use the `reasoning_effort` parameter to control depth instead."

---
---

## Session 2.3: Multimodal Models & Vision-Language Architecture

### 1. THE PROBLEM — Why Can't We Just Describe Images in Text?

Using OCR + image captioning → feeding text to an LLM is **lossy**. You can't describe a complex medical X-ray, circuit diagram, or dense chart with enough detail for accurate reasoning. We need a way to feed **raw pixel information** directly into the LLM as tokens.

---

### 2. THE NAIVE ALTERNATIVE — "Why Not Caption + Text LLM?"

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────┐
  │  📷 Image     │────▶│ Caption:      │────▶│ Text LLM  │
  │  (a chart     │     │ "A bar chart  │     │ answers   │
  │   showing     │     │  with 3 bars" │     │ about the │
  │   revenue)    │     │               │     │ "chart"   │
  └──────────────┘     └──────────────┘     └──────────┘

  What's LOST:
  ❌ Exact values on the bars (148.2M, 203.7M, 256.1M)
  ❌ Spatial relationships (which bar is tallest?)
  ❌ Colors and their meaning (red = loss, green = profit)
  ❌ Fine-grained details (small footnotes, axis labels)
  ❌ Nuance that's hard to describe in words
```

**What we need:** A way to feed the **raw pixel information** directly into the LLM as tokens it can attend to — just like text tokens.

---

### 3. HOW IMAGES BECOME TOKENS — Vision Transformer (ViT)

This is the core architecture. Step by step with real numbers:

#### Step 1: SPLIT IMAGE INTO PATCHES

```
  Original image: 224 × 224 pixels, 3 color channels (RGB)

  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ P8 │ P9 │P10 │P11 │P12 │P13 │P14 │
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  │P15 │P16 │    │    │    │    │    │    │    │    │    │    │    │P28 │
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  ├────┼────┤    ....                                        ├────┼────┤
  │    │    │                   14 × 14                      │    │    │
  :    :    :                  = 196 patches                 :    :    :
  │    │    │                                                │    │    │
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  │P183│P184│    │    │    │    │    │    │    │    │    │    │    │P196│
  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

  Each patch: 16 × 16 pixels × 3 channels = 768 values
  Total patches: (224/16) × (224/16) = 14 × 14 = 196 patches
```

#### Step 2: FLATTEN EACH PATCH INTO A VECTOR

```
  Patch P1 (16×16×3 = 768 raw pixel values):
  [0.23, 0.45, 0.67, 0.12, ..., 0.89]  ← 768-dim vector

  Patch P2:
  [0.91, 0.33, 0.55, 0.78, ..., 0.44]  ← 768-dim vector

  ...196 such vectors total
```

#### Step 3: LINEAR PROJECTION (Patch Embedding)

```
  Each 768-dim patch vector → projected to model dimension (e.g., 1024)

  Patch P1: [0.23, 0.45, ..., 0.89]  ──→  W_embed  ──→  [0.12, -0.34, ..., 0.56]
            768-dim                                       1024-dim
                                                          (same size as text tokens!)

  This is a LEARNABLE linear layer: E = patch × W_embed + b
```

#### Step 4: ADD POSITION EMBEDDINGS

```
  Just like text tokens need positional encoding,
  image patches need to know WHERE they are in the image!

  Patch P1 (top-left):       embedding + pos_1   → visual token 1
  Patch P2 (top-2nd):        embedding + pos_2   → visual token 2
  ...
  Patch P196 (bottom-right): embedding + pos_196 → visual token 196
```

#### Step 5: FEED THROUGH TRANSFORMER

```
  Now we have 196 "visual tokens" that look EXACTLY like text tokens
  to the transformer — same dimensionality, same format!

  [vis_1, vis_2, ..., vis_196] → Transformer Encoder → 196 output vectors
```

#### The Complete ViT Pipeline Diagram

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    VISION TRANSFORMER (ViT) PIPELINE                │
  │                                                                     │
  │  📷 Image (224×224×3)                                               │
  │       │                                                             │
  │       ▼                                                             │
  │  ┌─────────────────────┐                                            │
  │  │ Split into 16×16    │                                            │
  │  │ patches             │   196 patches, each 16×16×3 = 768 values  │
  │  └─────────┬───────────┘                                            │
  │            │                                                        │
  │            ▼                                                        │
  │  ┌─────────────────────┐                                            │
  │  │ Flatten each patch  │   196 vectors of dim 768                   │
  │  └─────────┬───────────┘                                            │
  │            │                                                        │
  │            ▼                                                        │
  │  ┌─────────────────────┐                                            │
  │  │ Linear Projection   │   768-dim → 1024-dim (learnable W_embed)  │
  │  │ (Patch Embedding)   │   Now SAME dimension as text tokens!      │
  │  └─────────┬───────────┘                                            │
  │            │                                                        │
  │            ▼                                                        │
  │  ┌─────────────────────┐                                            │
  │  │ + Position          │   Adds spatial location info              │
  │  │   Embeddings        │   (where in the image is this patch?)     │
  │  └─────────┬───────────┘                                            │
  │            │                                                        │
  │            ▼                                                        │
  │  ┌─────────────────────┐                                            │
  │  │ Transformer Encoder │   Self-attention across all 196 patches   │
  │  │ (multiple layers)   │   Patches "look at" each other            │
  │  └─────────┬───────────┘                                            │
  │            │                                                        │
  │            ▼                                                        │
  │  196 Visual Token Embeddings (dim 1024)                             │
  │  [vis_1, vis_2, ..., vis_196]                                       │
  │                                                                     │
  │  These are DIMENSIONALLY IDENTICAL to text token embeddings!        │
  └─────────────────────────────────────────────────────────────────────┘
```

**The KEY insight:** After ViT processing, an image is just a **sequence of token embeddings** — indistinguishable in format from text token embeddings. The transformer treats them the same way!

```
  Text tokens:   ["The", "cat", "sits", "on", "mat"]  → 5 embeddings of dim 1024
  Image tokens:  [P1, P2, P3, ..., P196]               → 196 embeddings of dim 1024
                                                         ↑
                                                    "Visual tokens"
                                            The LLM can't tell these apart!
```

> **Interview Tip 💡:** "A ViT converts images into a sequence of patch embeddings that are dimensionally identical to text token embeddings. This is what makes multimodal models possible — the LLM's self-attention mechanism doesn't know or care whether a token came from text or an image patch."

---

### 4. TWO MULTIMODAL ARCHITECTURES

Now the big architectural question: **How do you combine the vision encoder with the language model?**

#### Architecture 1: Adapter-Based (Bolt-on)

**Used by:** Early GPT-4V, LLaVA, many open-source models

```
  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
  │              │    │                │    │                  │
  │   Frozen     │    │   Projection   │    │   Frozen/Fine-   │
  │   Vision     │───▶│   Layer        │───▶│   tuned LLM      │
  │   Encoder    │    │   (adapter)    │    │                  │
  │   (ViT)      │    │                │    │   Text in +      │
  │              │    │   Maps vision  │    │   Visual tokens  │
  │  Trained     │    │   dim → LLM    │    │   → Text out     │
  │  separately  │    │   dim          │    │                  │
  └──────────────┘    └────────────────┘    └──────────────────┘
        ↑                    ↑                       ↑
   Pre-trained         Small trainable          Pre-trained
   (e.g., CLIP)        bridge layer             text LLM

  How it works:
  1. Vision encoder processes image → visual embeddings
  2. Adapter projects them to LLM's dimension
  3. Visual tokens are PREPENDED to text tokens
  4. LLM processes [visual tokens | text tokens] together

  Input to LLM:
  [img_1, img_2, ..., img_196, "What", "is", "in", "this", "image", "?"]
   ├──── visual tokens ────┤  ├──────── text tokens ─────────────────┤
```

- Vision encoder is typically **FROZEN** (pre-trained CLIP) — it **can't learn new visual concepts** for the language task
- Only the small adapter/projection layer is trained
- ✅ Pros: Easy to build, can use existing pre-trained models
- ❌ Cons: Vision and language don't deeply integrate. Image "understanding" is limited to what the ViT learned during its own pre-training


#### Architecture 2: Native Multimodal (End-to-End)

**Used by:** GPT-4o, Gemini models, Claude 3+

```
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │           SINGLE UNIFIED MODEL                         │
  │                                                        │
  │   Image pixels ─────┐                                  │
  │                      ├──→ Unified token space ──→ Out  │
  │   Text tokens  ─────┤    (trained together!)           │
  │                      │                                  │
  │   Audio waveform ───┘                                  │
  │                                                        │
  │   All modalities share the SAME transformer layers     │
  │   Trained on text + images + audio SIMULTANEOUSLY      │
  │                                                        │
  └────────────────────────────────────────────────────────┘

  ✅ Pros: Deep cross-modal understanding
           Can do things adapter models CAN'T:
           - Understand text IN images (signs, menus, code screenshots)
           - Reason about spatial relationships precisely
           - Generate images from the same model (some models)
  ❌ Cons: Extremely expensive to train
           Requires massive multimodal datasets
           Only big labs can build these
```

#### Detailed Comparison

| Dimension | Adapter-Based | Native Multimodal |
|-----------|--------------|-------------------|
| **Training** | Cheap — only train adapter layer | Very expensive — train entire model on all modalities |
| **Cross-modal understanding** | Shallow — vision encoder frozen, can't learn new visual concepts for language | Deep — vision and language representations intertwined |
| **Text-in-image** | Weak — ViT wasn't trained for OCR | Strong — trained on images with text |
| **Examples** | LLaVA, early GPT-4V | GPT-4o, Gemini 1.5/2.0, Claude 3/3.5/4 |
| **Open source** | Many available | Very few (Gemma, some Llama variants) |

> **Common Misconception ⚠️:** "Native multimodal" doesn't mean there's no vision encoder at all. GPT-4o and Gemini still use specialized visual processing components — but these components are **jointly trained** with the language model from scratch, not bolted on after the fact.

---

### 5. HIGH-RESOLUTION IMAGES — TILING STRATEGY

A standard ViT with 16×16 patches on a 224×224 image gives 196 tokens. But real images are much bigger! How do models handle 4K images?

```
  High-res image (2048 × 1024)
         │
         ▼
  ┌──────────────────────────────────────┐
  │  Split into tiles (e.g., 512×512)    │
  │                                      │
  │  ┌───────┬───────┬───────┬───────┐   │
  │  │       │       │       │       │   │
  │  │ Tile1 │ Tile2 │ Tile3 │ Tile4 │   │  = 4 tiles
  │  │       │       │       │       │   │
  │  ├───────┼───────┼───────┼───────┤   │
  │  │       │       │       │       │   │
  │  │ Tile5 │ Tile6 │ Tile7 │ Tile8 │   │  = 8 tiles total
  │  │       │       │       │       │   │
  │  └───────┴───────┴───────┴───────┘   │
  │                                      │
  │  + 1 thumbnail of the full image     │  = 9 total inputs to ViT
  └──────────────────────────────────────┘
         │
         ▼
  Each tile → ViT → ~196 tokens
  9 tiles × 196 = 1,764 visual tokens!

  ⚠️ This is why high-res image analysis is EXPENSIVE
     More tiles = more tokens = higher cost + slower
```

**Practical cost implications:**

| Image Resolution | Visual Tokens | Approx Cost |
|-----------------|---------------|-------------|
| 512 × 512 (low) | ~196 | ~$0.001 |
| 1024 × 1024 | ~784 | ~$0.004 |
| 2048 × 2048 | ~3,136 | ~$0.016 |
| 4096 × 4096 (4K) | ~12,000+ | ~$0.06+ |

```
Compare: 1 text token   ≈ $0.000003 (GPT-4o)
         1 image tile   ≈ equal to ~196 text tokens in cost
```

---

### 6. AUDIO & VOICE AI

Audio follows the **same principle** as vision — convert audio into token-like representations:

```
=== HOW AUDIO BECOMES TOKENS ===

  Audio waveform (e.g., 5 seconds of speech)
  ┌────────────────────────────────────────┐
  │  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿   │
  │  Raw audio: 16,000 samples/sec        │
  │  5 sec × 16,000 = 80,000 samples     │
  └─────────────────┬──────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  Step 1: Convert to spectrogram         │
  │  (frequency × time representation)      │
  │                                         │
  │  ┌─────────────────────────────────┐    │
  │  │ ████░░██░░████████░░██████░░█   │    │  ← Like a "picture"
  │  │ ██████████░░░░████████░░░░████  │    │     of the sound
  │  │ ░░████████████░░░░████████████  │    │
  │  └─────────────────────────────────┘    │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  Step 2: Audio encoder (like ViT but     │
  │  for spectrograms) → audio tokens       │
  │                                         │
  │  5 sec speech → ~50-100 audio tokens    │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  Step 3: Feed to LLM alongside text     │
  │                                         │
  │  [audio_1, ..., audio_50, "Transcribe"] │
  │                                         │
  │  LLM processes audio tokens + text      │
  │  tokens together with self-attention    │
  └─────────────────────────────────────────┘
```

#### Two Approaches to Voice AI

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  CASCADE (Traditional)               │  NATIVE (GPT-4o style)              │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  Audio → ASR → Text → LLM → Text   │  Audio → ┌───────────┐ → Audio     │
│              → TTS → Audio          │          │ UNIFIED   │             │
│                                     │          │ MODEL     │             │
│  Speech─to─text─to─speech           │          │           │             │
│  3 separate models                  │          └───────────┘             │
│                                     │  Audio in, audio out directly      │
│  Latency: 2-5 seconds              │  Latency: 200-500ms               │
│  Loses: tone, emotion, accent      │  Preserves: tone, emotion, accent │
│  Can't: sing, do accents, whisper  │  Can: laugh, whisper, sing        │
│                                     │                                     │
│  Example: Alexa, old Siri          │  Example: GPT-4o voice mode,       │
│                                     │  Gemini Live                        │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

#### Voice AI Comparison Table

| Dimension | Cascade (Traditional) | Native (Modern) |
|-----------|----------------------|-----------------|
| **Pipeline** | Audio → ASR → Text → LLM → Text → TTS → Audio | Audio → Unified Model → Audio |
| **Models in pipeline** | 3 separate models | 1 model |
| **Latency** | 2-5 seconds | 200-500ms |
| **Preserves tone/emotion** | No — lost in text conversion | Yes |
| **Can laugh/whisper/sing** | No | Yes |
| **Examples** | Old Siri, Alexa | GPT-4o voice, Gemini Live |

> **Interview Tip 💡:** "Native audio models like GPT-4o's voice mode process audio tokens directly — they don't go through speech-to-text first. This preserves paralinguistic features like tone, emotion, and pacing, and dramatically reduces latency from seconds to hundreds of milliseconds. It's the same principle as native vision — all modalities share the transformer's attention layers."

---

### 7. 2026 MAJOR MULTIMODAL MODELS

| Model | Provider | Modalities | Architecture | Standout Feature |
|-------|----------|-----------|-------------|------------------|
| **GPT-4o** | OpenAI | Text + Image + Audio + Video | Native multimodal | Real-time voice conversation, audio in/out |
| **o3/o4-mini** | OpenAI | Text + Image | Reasoning + vision | Can reason about images with hidden CoT |
| **Gemini 2.5 Pro** | Google | Text + Image + Audio + Video + Code | Native multimodal | 1M+ token context, natively understands video |
| **Gemini 2.0 Flash** | Google | Text + Image + Audio | Native, optimized for speed | Fast multimodal at low cost |
| **Claude 4 Opus/Sonnet** | Anthropic | Text + Image | Native vision, extended thinking | Strong document/chart analysis, PDF understanding |
| **Llama 3.2 Vision** | Meta | Text + Image | Adapter-based (open source) | Best open-source multimodal |
| **Phi-4 multimodal** | Microsoft | Text + Image | Small native multimodal | Runs on device, strong for size |

---

### 8. INTERVIEW ANGLES

> **Q: "How does a multimodal model 'see' an image?"**
>
> **Model Answer:** "The image is split into fixed-size patches (typically 16×16), each patch is flattened and projected through a learnable linear layer to match the text embedding dimension. Positional embeddings are added, and the result is a sequence of visual tokens that are dimensionally identical to text tokens — the transformer processes them with the same self-attention mechanism, not knowing or caring whether a token came from text or pixels."

> **Q: "What's the difference between adapter-based and native multimodal models?"**
>
> **Model Answer:** "Adapter-based models (like LLaVA) bolt a frozen pre-trained vision encoder (e.g., CLIP) onto a pre-trained LLM via a small projection layer. Only the adapter is trained, so it's cheap but gives shallow cross-modal understanding — the frozen ViT can't learn new visual concepts. Native multimodal models (like GPT-4o, Gemini) train vision and language components together end-to-end, creating deeply intertwined representations. This enables things adapter models struggle with: reading text in images, precise chart analysis, and spatial reasoning."

> **Q: "Why is high-resolution image analysis expensive for LLMs?"**
>
> **Model Answer:** "High-res images are split into multiple tiles, each processed by the vision encoder into ~196 tokens. A 2048×2048 image might produce 3,000+ visual tokens — more than many text prompts. Since LLM cost scales with token count, high-res vision tasks can be 10-60× more expensive than text-only queries. This is why production systems often downsample images unless fine detail is needed."

> **Q: "What advantage does native voice AI (like GPT-4o) have over cascade speech systems?"**
>
> **Model Answer:** "Native voice models process audio tokens directly without speech-to-text conversion. This preserves paralinguistic features — tone, emotion, pacing, accent — that are lost when speech is converted to text. It also reduces latency from 2-5 seconds to 200-500ms, enabling real-time conversation. The model can even generate non-speech audio cues like laughter or whispering."

---
---

## Session 2.4: The 2026 Model Landscape & Model Routing

### 1. THE PROBLEM — Why Not Just Use the Best Model for Everything?

```
  If you use Claude 4 Opus for EVERY task:

  Task: "Is this email spam?"
  Model: Claude 4 Opus
  Cost: $0.05 per query
  
  At 100K emails/day: $5,000/day = $150,000/month

  With a fine-tuned SLM:
  Cost: $0.0005 per query
  At 100K emails/day: $50/day = $1,500/month

  You're paying 100× MORE for ZERO quality improvement!
```

**The real skill is knowing WHICH model to use for WHICH task.** Interviewers test this — not just "name the best model" but "design a cost-effective model selection strategy."

---

### 2. THE 5-TIER MODEL LANDSCAPE

```
┌──────────────────────────────────────────────────────────────────────┐
│                    THE 2026 MODEL LANDSCAPE                          │
│                                                                      │
│  TIER 1: FRONTIER REASONING    (The "Deep Thinkers")                │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  o3, o4-mini                          (OpenAI)         │         │
│  │  Claude 4 Opus + Extended Thinking    (Anthropic)      │         │
│  │  Gemini 2.5 Pro                       (Google)         │         │
│  │                                                        │         │
│  │  ✅ Best for: Math, logic, complex code, research      │         │
│  │  💰 Cost: $$$$ ($10-75 / 1M tokens)                   │         │
│  │  ⏱️ Latency: 5-120 seconds                            │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                      │
│  TIER 2: FRONTIER STANDARD     (The "Workhorses")                   │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  GPT-4o                               (OpenAI)         │         │
│  │  Claude 4 Sonnet                      (Anthropic)      │         │
│  │  Gemini 2.5 Flash                     (Google)         │         │
│  │                                                        │         │
│  │  ✅ Best for: General tasks, chat, content, code gen    │         │
│  │  💰 Cost: $$ ($2-10 / 1M tokens)                      │         │
│  │  ⏱️ Latency: 0.5-3 seconds                            │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                      │
│  TIER 3: MID-TIER / FAST       (The "Speed Demons")                 │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  GPT-4o-mini                          (OpenAI)         │         │
│  │  Claude 4 Haiku                       (Anthropic)      │         │
│  │  Gemini 2.0 Flash                     (Google)         │         │
│  │                                                        │         │
│  │  ✅ Best for: High-volume, routing, classification      │         │
│  │  💰 Cost: $ ($0.10-1 / 1M tokens)                     │         │
│  │  ⏱️ Latency: 0.1-1 second                             │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                      │
│  TIER 4: SMALL LANGUAGE MODELS (The "Edge Runners")                 │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  Phi-4 (3.8B)                         (Microsoft)      │         │
│  │  Gemma 3 (2B, 4B, 12B)               (Google)         │         │
│  │  Llama 3.2 (1B, 3B)                  (Meta)           │         │
│  │  Qwen 2.5 (0.5B-7B)                  (Alibaba)        │         │
│  │                                                        │         │
│  │  ✅ Best for: On-device, offline, privacy-sensitive     │         │
│  │  💰 Cost: ¢ (self-hosted, near-zero marginal cost)    │         │
│  │  ⏱️ Latency: 10-100ms (on device)                     │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                      │
│  TIER 5: SPECIALIZED MODELS    (The "Specialists")                  │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  Embedding models: text-embedding-3, Voyage 3          │         │
│  │  Rerankers: Cohere Rerank, bge-reranker                │         │
│  │  Code models: Codex, DeepSeek Coder V3                │         │
│  │  Image gen: DALL-E 3, Imagen 3, Flux                   │         │
│  │  Speech: Whisper, ElevenLabs                           │         │
│  │                                                        │         │
│  │  ✅ Best for: Specific tasks they're designed for       │         │
│  │  💰 Cost: Varies                                       │         │
│  └────────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3. PROVIDER COMPARISON — The Big Three + Open Source

```
┌─────────────┬──────────────────┬──────────────────┬──────────────────┬───────────────────┐
│ Dimension   │ OpenAI           │ Anthropic        │ Google           │ Open Source       │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Top         │ o3               │ Claude 4 Opus    │ Gemini 2.5 Pro   │ Llama 3.3 70B    │
│ Reasoning   │ o4-mini          │ + ext. thinking  │                  │ DeepSeek R1      │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Top         │ GPT-4o           │ Claude 4 Sonnet  │ Gemini 2.5 Flash │ Llama 3.1 405B   │
│ Standard    │                  │                  │                  │ Mixtral 8x22B    │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Fast/Cheap  │ GPT-4o-mini      │ Claude 4 Haiku   │ Gemini 2.0 Flash │ Llama 3.2 3B     │
│             │                  │                  │                  │ Phi-4, Gemma 3   │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Multimodal  │ ✅ Text+Img+     │ ✅ Text+Img      │ ✅ Text+Img+     │ ⚠️ Limited       │
│             │ Audio+Video      │                  │ Audio+Video      │ (LLaVA, Llama    │
│             │                  │                  │                  │  3.2 Vision)     │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Context     │ 128K             │ 200K             │ 1M-2M            │ 8K-128K          │
│ Window      │                  │                  │ (largest!)       │ (varies)         │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Strength    │ Best reasoning   │ Best code +      │ Best multimodal  │ Full control,    │
│             │ models, tools    │ safety, long     │ + context +      │ privacy, no      │
│             │ ecosystem        │ documents        │ cost efficiency  │ vendor lock-in   │
├─────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ Weakness    │ Expensive,       │ No audio/video,  │ Historically     │ Smaller, less    │
│             │ closed source    │ closed source    │ less reliable    │ capable, need    │
│             │                  │                  │ on complex tasks │ infra to host    │
└─────────────┴──────────────────┴──────────────────┴──────────────────┴───────────────────┘
```

---

### 4. THE 5-QUESTION MODEL SELECTION FRAMEWORK

Ask these **in order** before choosing a model:

```
┌──────────────────────────────────────────────────────────────────────┐
│  QUESTION 1: Does this need multi-step REASONING?                    │
│                                                                      │
│    YES → Reasoning model (o3, Gemini 2.5 Pro, Claude ext thinking)  │
│    NO  → Continue to Q2                                              │
│                                                                      │
│  QUESTION 2: What's the QUALITY bar?                                 │
│                                                                      │
│    "Must be perfect" → Frontier (GPT-4o, Claude Sonnet)             │
│    "Good enough"     → Mid-tier (GPT-4o-mini, Haiku, Flash)         │
│    "Just classify"   → SLM or fine-tuned small model                 │
│                                                                      │
│  QUESTION 3: What's the VOLUME?                                      │
│                                                                      │
│    <1K queries/day   → Cost doesn't matter, use best model           │
│    1K-100K/day       → Cost matters, use tiered routing              │
│    >100K/day         → Cost is CRITICAL, SLM + fine-tuning          │
│                                                                      │
│  QUESTION 4: What's the LATENCY requirement?                         │
│                                                                      │
│    Real-time (<500ms)  → SLM on device or cached responses           │
│    Interactive (<3s)   → Mid-tier or frontier standard               │
│    Batch (don't care)  → Cheapest option that meets quality          │
│                                                                      │
│  QUESTION 5: Any SPECIAL requirements?                               │
│                                                                      │
│    Privacy/offline   → Open source SLM on-device                     │
│    Image/audio input → Native multimodal (GPT-4o, Gemini)           │
│    Very long docs    → Gemini (1M+ context) or chunked RAG          │
│    Tool calling      → GPT-4o or Claude (best function calling)     │
│    Safety-critical   → Claude (strongest safety, least jailbreak)    │
└──────────────────────────────────────────────────────────────────────┘
```

#### Decision Tree Diagram

```
                    MODEL SELECTION DECISION TREE

                         ┌──────────────┐
                         │  START HERE   │
                         │  What's the   │
                         │  task?        │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ REASONING?   │  │ GENERATION?  │  │ PROCESSING?  │
     │ (math, logic,│  │ (write, code,│  │ (classify,   │
     │  planning,   │  │  create,     │  │  extract,    │
     │  debugging)  │  │  translate)  │  │  summarize)  │
     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
            │                 │                 │
            ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │How complex? │   │Quality bar? │   │   Volume?    │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
     ╱    │    ╲        ╱    │    ╲        ╱    │    ╲
   Low   Med  High   Low   Med  High   Low   Med  High
    │     │     │      │     │     │      │     │     │
    ▼     ▼     ▼      ▼     ▼     ▼      ▼     ▼     ▼
  Std   Rsn   Rsn    Mini  Std  Front  Front  Mid   SLM/
  LLM   med   high   /Mid  LLM   ier   ier   tier  Fine-
        eff   eff                                    tuned
```

---

### 5. COST COMPARISON — Real Numbers

```
┌──────────────────┬────────────┬────────────┬────────────┬───────────┐
│                  │ Input Cost │ Output Cost│ 1K queries │ 1M queries│
│ Model            │ /1M tokens │ /1M tokens │ (avg 1K    │ per month │
│                  │            │            │  tok each) │           │
├──────────────────┼────────────┼────────────┼────────────┼───────────┤
│ o3 (high)        │ $10.00     │ $40.00     │ $50.00     │ $50,000   │
│ GPT-4o           │ $2.50      │ $10.00     │ $12.50     │ $12,500   │
│ Claude 4 Sonnet  │ $3.00      │ $15.00     │ $18.00     │ $18,000   │
│ Gemini 2.5 Flash │ $0.15      │ $0.60      │ $0.75      │ $750      │
│ GPT-4o-mini      │ $0.15      │ $0.60      │ $0.75      │ $750      │
│ Claude 4 Haiku   │ $0.25      │ $1.25      │ $1.50      │ $1,500    │
│ Llama 3.2 3B     │ ~$0.01*    │ ~$0.01*    │ $0.02      │ $20       │
│ (self-hosted)    │            │            │            │           │
└──────────────────┴────────────┴────────────┴────────────┴───────────┘
  * Self-hosted cost = GPU rental amortized per token

  KEY INSIGHT: The cheapest frontier model is ~500× more expensive
               than a self-hosted SLM. At scale, this is the
               difference between $750/month and $50,000/month.
```

---

### 6. REAL-WORLD ROUTING SCENARIOS

#### Scenario 1: E-Commerce Product Search

```
  Task: "blue running shoes under $100" → retrieve, rank, show
  Volume: 500K searches/day | Latency: <200ms | Quality: "Good enough"

  Answer: SLM or fine-tuned small model (Tier 4)
  Why: Massive volume + strict latency. This is RETRIEVAL, not reasoning.
       A fine-tuned embedding model + reranker is ideal. No LLM needed
       for most queries — use traditional search + ML ranking.
```

#### Scenario 2: Legal Contract Analysis

```
  Task: Review 50-page contract, identify risky clauses
  Volume: 200 docs/day | Latency: 30-60s OK | Quality: MUST be accurate

  Answer: Frontier Reasoning (Tier 1) — o3 or Gemini 2.5 Pro
  Why: Low volume (cost OK), accuracy critical, needs multi-step reasoning.
       Gemini 2.5 Pro with 1M context handles full document.
```

#### Scenario 3: Customer Support Chatbot

```
  Task: Mix of FAQs, account issues, billing disputes
  Volume: 10K queries/day | Latency: <2s | Quality: Mixed

  Answer: TIERED ROUTING

  ┌──────────────────────────────────────────────────┐
  │  Query arrives                                    │
  │       │                                           │
  │       ▼                                           │
  │  ┌──────────┐                                     │
  │  │ ROUTER   │  ← Tier 3 (GPT-4o-mini or Haiku)  │
  │  │ Classify │     Classifies query complexity      │
  │  └────┬─────┘                                     │
  │    ┌──┼──────┐                                    │
  │    │  │      │                                    │
  │    ▼  ▼      ▼                                    │
  │  FAQ  Gen   Complex                               │
  │  70%  20%   10%                                   │
  │    │   │      │                                   │
  │    ▼   ▼      ▼                                   │
  │  RAG  Tier2  Tier1                                │
  │  +T4  Std   Reasoning                             │
  │       LLM   + Human                               │
  │             escalation                             │
  └──────────────────────────────────────────────────┘
```

#### Scenario 4: AI Code Assistant (IDE)

```
  Task: Autocomplete, explain, refactor, debug
  Volume: 1000s of completions/dev/day | Latency: varies

  │ Feature          │ Model                │ Why              │
  │ Autocomplete     │ SLM (Phi-4, Gemma)   │ Speed critical   │
  │ Code explanation │ Mid-tier (4o-mini)    │ Fast + adequate  │
  │ Refactoring      │ Frontier (Claude 4)   │ Quality matters  │
  │ Complex debug    │ Reasoning (o3)        │ Deep analysis    │
```

---

### 7. THE MODEL ROUTER ARCHITECTURE

This is the pattern we'll BUILD in Session 2.5:

```
┌──────────────────────────────────────────────────────────────────────┐
│                INTELLIGENT MODEL ROUTER ARCHITECTURE                 │
│                                                                      │
│  User Query                                                          │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────┐                            │
│  │         QUERY ANALYZER               │                            │
│  │  (Lightweight — SLM or rule-based)   │                            │
│  │                                      │                            │
│  │  Extracts:                           │                            │
│  │  • Task type (reasoning/generation/  │                            │
│  │    classification/retrieval)         │                            │
│  │  • Complexity (simple/medium/hard)   │                            │
│  │  • Modality (text/image/audio)       │                            │
│  │  • Latency requirement               │                            │
│  │  • Domain (code/legal/medical/gen)   │                            │
│  └──────────────┬───────────────────────┘                            │
│                 │                                                     │
│                 ▼                                                     │
│  ┌──────────────────────────────────────┐                            │
│  │         ROUTING LOGIC                 │                            │
│  │                                      │                            │
│  │  Rules + ML classifier:              │                            │
│  │  IF reasoning + hard → Tier 1        │                            │
│  │  IF generation + medium → Tier 2     │                            │
│  │  IF classification + any → Tier 3/4  │                            │
│  │  IF multimodal → ensure vision model │                            │
│  │  IF latency < 500ms → Tier 3/4 only │                            │
│  └──────────────┬───────────────────────┘                            │
│                 │                                                     │
│        ┌────────┼────────┬────────┐                                  │
│        │        │        │        │                                   │
│        ▼        ▼        ▼        ▼                                   │
│  ┌─────────┐ ┌────────┐ ┌──────┐ ┌──────┐                           │
│  │ Tier 1  │ │ Tier 2 │ │Tier 3│ │Tier 4│                           │
│  │Reasoning│ │Frontend│ │ Fast │ │ SLM  │                           │
│  │ o3      │ │GPT-4o  │ │4o-min│ │Phi-4 │                           │
│  └────┬────┘ └───┬────┘ └──┬───┘ └──┬───┘                           │
│       │          │         │        │                                │
│       └──────────┴─────────┴────────┘                                │
│                      │                                               │
│                      ▼                                               │
│  ┌──────────────────────────────────────┐                            │
│  │         RESPONSE + METADATA          │                            │
│  │                                      │                            │
│  │  • Model used                        │                            │
│  │  • Tokens consumed                   │                            │
│  │  • Latency                           │                            │
│  │  • Cost                              │                            │
│  │  • Confidence score                  │                            │
│  │                                      │                            │
│  │  (logged for optimization)           │                            │
│  └──────────────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
```

> **Interview Tip 💡:** "A model router is essential in production. The router itself must be cheap and fast — use rules + a lightweight classifier, never a frontier model. Log every routing decision with cost/latency/quality metrics so you can continuously optimize. This is similar to how CDNs route traffic — send requests to the cheapest endpoint that meets the SLA."

---

### 8. INTERVIEW ANGLES

> **Q: "How would you design model selection for a production system?"**
>
> **Model Answer:** "I'd use a 5-question framework: (1) Does it need reasoning? (2) What's the quality bar? (3) What's the volume? (4) What's the latency requirement? (5) Any special needs (privacy, multimodal, long context)? Based on answers, I route to the appropriate tier. For mixed-complexity systems, I use a lightweight router (SLM or rules) that classifies each query and sends it to the cheapest model that meets the SLA. I always log routing decisions with cost/latency/quality metrics for continuous optimization."

> **Q: "What are the key trade-offs between model providers in 2026?"**
>
> **Model Answer:** "OpenAI leads in reasoning (o3/o4) and tool ecosystem. Anthropic excels at code, safety, and long-document analysis (200K context). Google's Gemini offers the best multimodal capabilities and largest context window (1M+) at competitive pricing. Open-source models (Llama, Phi, Gemma) provide full control and privacy but require infrastructure to host and are less capable. The trend is convergence — all providers are improving across dimensions — so the choice often comes down to specific strengths matching your use case."

> **Q: "Walk me through how you'd design document processing for a bank (50K docs/day)."**
>
> **Model Answer:** "Three-tier routing with a lightweight classifier. Simple forms (70%, ~35K/day) go through OCR + template extraction — many don't need an LLM, keeping cost near zero. Standard contracts (20%, ~10K/day) go to a frontier standard model like Claude Sonnet for accurate clause extraction. Complex regulatory filings (10%, ~5K/day) go to a reasoning model for multi-step analysis. Since this is batch processing, latency is flexible so I optimize for cost-quality. Total ~$1,600/day vs $15,000/day without routing. For a bank, I add confidence scoring and route low-confidence regulatory outputs to human reviewers — legal liability demands it."
