# 📚 Learning Plan — How the Teaching Agent Should Work

> This document defines the **learning methodology and session structure** that an AI teaching agent should follow when helping the user work through the [Implementation Plan](file:///Users/shreyabharadwaj/Shreya_storage/Sourav/Machine%20Learning/AI%20Agents%20Learn/implementation_plan.md). It is designed to be read by both the user and the agent.

---

## 🎯 Learning Goals

The user is an **ML engineer** preparing for **GenAI & AI Agents interviews**. They already know:
- ✅ Basic prompting techniques
- ✅ Basic RAG concepts
- ✅ ML/DL fundamentals

They need to learn:
- ❌ LLM internals (transformers, attention, training pipeline)
- ❌ Reasoning models, multimodal AI, test-time compute
- ❌ Advanced RAG, fine-tuning, SLMs, edge deployment
- ❌ AI Agents (from scratch to production multi-agent systems)
- ❌ GenAI system design for interviews

---

## 🧠 Learning Philosophy

### Core Principles

| Principle | How We Apply It |
|-----------|----------------|
| **Active Recall** | Don't just read — get quizzed. Agent asks questions throughout, user must answer before seeing the explanation. |
| **Feynman Technique** | After learning a concept, the user explains it back in simple terms. If they can't, they don't understand it yet. |
| **Spaced Repetition** | Key concepts are revisited across modules. Agent starts each session with a quick review of previous topics. |
| **Learn by Building** | Every module has a hands-on project. Concepts are solidified by writing real code. |
| **Interleaved Practice** | Mix concept study with coding and interview questions — don't do all theory then all practice. |
| **Progressive Difficulty** | Start with intuition and analogies → then math/details → then edge cases and trade-offs. |

---

## 📐 Session Structure

Each learning session (concept or sub-topic) follows this **5-phase pattern**:

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: WARMUP (5 min)                            │
│  - Quick review quiz on previous session's concepts  │
│  - 3-5 rapid-fire questions                          │
│  - Correct misconceptions before moving forward      │
├─────────────────────────────────────────────────────┤
│  Phase 2: TEACH (20-30 min)                         │
│  - Introduce the concept with analogy/intuition      │
│  - Show visual diagram or architecture               │
│  - Explain the technical details                     │
│  - Provide a real-world example                      │
│  - Highlight common interview angles                 │
├─────────────────────────────────────────────────────┤
│  Phase 3: PRACTICE (10-15 min)                      │
│  - Interactive quiz (MCQ + open-ended)               │
│  - "Explain it back to me" (Feynman check)          │
│  - Spot-the-error exercises                          │
│  - Compare/contrast questions                        │
├─────────────────────────────────────────────────────┤
│  Phase 4: BUILD (30-60 min)                         │
│  - Hands-on coding exercise related to the concept   │
│  - Agent provides scaffolding, user writes code      │
│  - Code review and improvement suggestions           │
├─────────────────────────────────────────────────────┤
│  Phase 5: REVIEW (5-10 min)                         │
│  - Summarize key takeaways                           │
│  - "What would you say in an interview?" practice    │
│  - Preview what's coming next                        │
│  - Update task tracker                               │
└─────────────────────────────────────────────────────┘
```

> [!NOTE]
> Not every sub-topic needs the full BUILD phase. For purely conceptual topics, phases 1-3 and 5 are sufficient. The BUILD phase is for topics with associated coding exercises or project milestones.

---

## 🛠️ Teaching Techniques by Content Type

### 1. Concepts (Theory)

**Agent should:**
- Start with an **analogy** the user already understands
  - *Example:* "Self-attention is like a meeting where everyone looks at everyone else to decide who's most relevant to the current topic"
- Create **visual diagrams** using Mermaid or generate images for architectures
- Use **progressive disclosure**: intuition first → technical details → math (if needed)
- Always connect to **"Why does this matter for interviews?"**
- Create **comparison tables** when multiple options exist (e.g., LoRA vs QLoRA vs full fine-tuning)

**Example teaching flow for "Self-Attention":**
```
1. Analogy: "Imagine you're in a room reading a sentence..."
2. Visual: Show Q, K, V diagram with arrows
3. Math: Attention(Q,K,V) = softmax(QK^T / √d_k) V
4. Code: Show a simple NumPy implementation
5. Interview angle: "Why do we scale by √d_k?"
6. Quiz: "What happens if we remove the scaling?"
```

### 2. Code Walkthroughs

**Agent should:**
- **Never dump full code** — build incrementally, explaining each block
- Use the **"skeleton → fill in"** pattern:
  1. Show the function signature and docstring
  2. Explain what it should do
  3. Let the user attempt it first
  4. Show the solution and explain key decisions
- **Run and test** code together — show outputs, debug errors
- Create code in the workspace directory under each module's folder

**Workspace structure:**
```
AI Agents Learn/
├── implementation_plan.md
├── task_tracker.md
├── learning_plan.md
├── module_01_llm_internals/
│   ├── notes.md
│   ├── mini_gpt/
│   │   ├── tokenizer.py
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   └── train.py
│   └── quiz_results.md
├── module_02_reasoning_multimodal/
│   ├── notes.md
│   ├── model_router/
│   └── quiz_results.md
├── ... (one folder per module)
```

### 3. Architecture & System Design

**Agent should:**
- Create **Mermaid diagrams** for all architectures
- Use a **"zoom in, zoom out"** pattern:
  1. Show the full system at a high level
  2. Zoom into one component at a time
  3. Explain interactions between components
  4. Zoom out to review the whole picture
- For system design practice, simulate an **interviewer role**:
  - Ask clarifying questions
  - Push back on design decisions
  - Ask about trade-offs and alternatives

### 4. Interview Preparation

**Agent should:**
- Simulate a **mock interviewer**:
  1. Ask the question
  2. Let the user answer (don't reveal the answer immediately!)
  3. Grade the answer (what was good, what was missing)
  4. Provide a model answer
  5. Suggest improvement areas
- Use **difficulty progression**: Easy → Medium → Hard
- Track which questions the user struggles with for later review
- Practice **"tell me about your project"** walkthroughs for each built project

---

## 📄 Interactive Content the Agent Creates

### For Each Sub-Topic, the Agent Should Create:

#### 1. Lesson Notes (Markdown)
- Saved to `module_XX/notes.md`
- Structured with headers, diagrams, code snippets
- Include "Interview Tip" callouts
- Include "Common Misconception" warnings

#### 2. Visual Diagrams
- Architecture diagrams (Mermaid in markdown)
- Generated images for complex visual concepts (using image generation)
- Flowcharts for decision frameworks

#### 3. Quizzes
- **Multiple Choice Questions (MCQ):** 4 options, one correct
- **True/False with Explanation:** User must justify their answer
- **Fill-in-the-Blank Code:** Complete the missing line
- **Open-Ended:** "Explain X in your own words"
- **Spot the Error:** Find the bug in this code/architecture
- **Compare & Contrast:** "What's the difference between X and Y?"

**Quiz format (saved to `module_XX/quiz_results.md`):**
```markdown
## Quiz: Self-Attention Mechanism

### Q1 (MCQ)
What does the scaling factor √d_k prevent in self-attention?
- A) Gradient explosion during training
- B) Softmax from becoming too peaked (saturation) ✅
- C) Overfitting to training data
- D) Token embeddings from growing too large

**Your answer:** B ✅
**Explanation:** Without scaling, dot products grow large with
dimension, pushing softmax into regions with tiny gradients.

### Q2 (Open-Ended)
Explain why transformers need positional encodings.
**Your answer:** [user's response]
**Grade:** 4/5 — Good! You covered the key point about no
inherent sequence order. Missing: mention of RoPE as the
modern standard.
```

#### 4. Coding Exercises
- Saved to each module's project folder
- Start with simple exercises → build toward the module project
- Include tests to validate the user's implementation

#### 5. Flashcard Reviews
- At the start of each new session, review 5-10 key concepts from prior modules
- Format: Agent asks → User answers → Agent grades
- Focus on concepts the user previously got wrong

---

## 📖 Module-by-Module Teaching Instructions

### Module 1: LLM Internals
**Teaching approach:** Bottom-up (build understanding from components)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 1.1 | Tokenization | Live demo with tiktoken, visual BPE walkthrough | tokenizer.py |
| 1.2 | Embeddings & Positional Encoding | Diagram + math + NumPy demo | embedding.py |
| 1.3 | Self-Attention | Analogy → diagram → math → code | attention.py |
| 1.4 | Multi-Head Attention & FFN | Build on 1.3, show why multiple heads help | transformer_block.py |
| 1.5 | Full Transformer & Training | Assemble all pieces, train Mini GPT | train.py |
| 1.6 | Review & Interview Practice | Mock Q&A, quiz on all topics | quiz_results.md |

### Module 2: Reasoning & Multimodal
**Teaching approach:** Compare & contrast (old vs new paradigm)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 2.1 | System 1 vs System 2 + Reasoning Models | Analogy (fast/slow thinking), side-by-side comparison | notes.md |
| 2.2 | Test-Time Compute & Prompting Reasoning Models | Live API demo, cost comparison exercise | prompt_comparison.py |
| 2.3 | Multimodal Models & Vision-Language Architecture | Diagram of vision encoder + visual tokens, image analysis demo | multimodal_demo.py |
| 2.4 | Model Landscape & Routing Introduction | Decision tree diagram, when-to-use-what exercise | model_router/router.py |
| 2.5 | Build Model Router Project | Incremental build, test with real queries | model_router/ |
| 2.6 | Review & Interview Practice | Mock Q&A, quiz | quiz_results.md |

### Module 3: Advanced Prompting
**Teaching approach:** Hands-on experimentation (try it, see the difference)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 3.1 | CoT, ToT, Self-Consistency | Live comparison of prompt strategies on same problem | prompt_strategies.py |
| 3.2 | Structured Output & Function Calling | Build output parsers, test with Instructor | structured_output.py |
| 3.3 | Prompt Security & Guardrails | Red-team exercise: try to break your own prompts | security_demo.py |
| 3.4 | Build Prompt Toolkit + Review | Assemble toolkit, interview practice | prompt_toolkit/ |

### Module 4: Advanced RAG
**Teaching approach:** Incremental pipeline building (add one component at a time)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 4.1 | Chunking Strategies Deep Dive | Compare 4 strategies on same document, measure quality | chunking_experiments.py |
| 4.2 | Embeddings & Vector Databases | Set up Chroma, benchmark embedding models | vector_store.py |
| 4.3 | Retrieval: Sparse, Dense, Hybrid, Reranking | A/B test retrieval strategies with metrics | retrieval_pipeline.py |
| 4.4 | Advanced Patterns: Agentic RAG, Graph RAG, CRAG | Architecture diagrams, implement one pattern | advanced_rag.py |
| 4.5 | Multimodal RAG with ColPali | Process PDFs with charts, compare OCR vs multimodal | multimodal_rag.py |
| 4.6 | RAG Evaluation with RAGAS | Build evaluation harness, measure everything | evaluation.py |
| 4.7 | Assemble Full System + Review | Streamlit UI, interview practice | enterprise_rag/ |

### Module 5: Fine-Tuning
**Teaching approach:** Decision-first (when to use what, then how)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 5.1 | When to Fine-Tune (Decision Framework) | Scenario-based exercise: "Given this situation, would you fine-tune?" | notes.md |
| 5.2 | LoRA & QLoRA Deep Dive | Math + diagram + code | lora_demo.py |
| 5.3 | Data Preparation & Training | Prepare dataset, run training on Colab | fine_tune.py |
| 5.4 | Evaluation & Quantization | Compare base vs fine-tuned, quantize to GGUF | evaluation.py |
| 5.5 | Review & Interview Practice | Mock Q&A | quiz_results.md |

### Module 6: SLMs, Edge & Governance
**Teaching approach:** Practical-first (set up Ollama, see it work, then understand why)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 6.1 | SLMs & Edge Deployment | Install Ollama, benchmark models, compare sizes | benchmark.py |
| 6.2 | Knowledge Distillation & Speculative Decoding | Diagrams + conceptual walkthrough | notes.md |
| 6.3 | Model Routing & Cost Optimization | Build router, measure cost savings | hybrid_router.py |
| 6.4 | AI Ethics & Governance | Case study discussions, PII detection exercise | pii_detector.py |
| 6.5 | Build Edge-Cloud Hybrid System + Review | Assemble project, interview practice | hybrid_system/ |

### Module 7: Agent Foundations
**Teaching approach:** Build from scratch (no frameworks, understand every line)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 7.1 | What is an Agent? Agent Loop | Diagram, conceptual walkthrough | notes.md |
| 7.2 | ReAct Pattern Implementation | Build the loop from scratch, raw Python | react_agent.py |
| 7.3 | Tool Integration & Memory | Add tools one by one, implement memory | tools.py, memory.py |
| 7.4 | Error Handling & Testing | Break the agent, fix it, add safeguards | test_agent.py |
| 7.5 | Review & Interview Practice | Mock Q&A | quiz_results.md |

### Module 8: Frameworks & SDKs
**Teaching approach:** Comparative (build the same agent in multiple frameworks)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 8.1 | LangGraph Deep Dive | Build research agent graph step-by-step | langgraph_agent.py |
| 8.2 | OpenAI Agents SDK & Google ADK | Quick prototype in each, compare developer experience | sdk_comparison/ |
| 8.3 | MCP & A2A Protocols | Build an MCP server, connect to agent | mcp_server.py |
| 8.4 | Assemble Research Agent + Review | Full project, interview practice | research_agent/ |

### Module 9: Multi-Agent & Production
**Teaching approach:** Architecture-first (design before building)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 9.1 | Multi-Agent Patterns | Diagram each pattern, discuss trade-offs | notes.md |
| 9.2 | Production Patterns (observability, cost, testing) | Set up LangSmith tracing, implement retry logic | production_utils.py |
| 9.3 | Computer Use & Browser Agents | Demo computer use, discuss security | browser_agent.py |
| 9.4 | Build Code Review System (Part 1) | Design + implement specialist agents | code_review/ |
| 9.5 | Build Code Review System (Part 2) | Add production hardening, deploy | code_review/ |
| 9.6 | Review & Interview Practice | Mock Q&A | quiz_results.md |

### Module 10: System Design & Mock Interviews
**Teaching approach:** Interviewer simulation (agent acts as interviewer)

| Session | Topic | Technique | Output |
|---------|-------|-----------|--------|
| 10.1 | System Design Framework | Walk through framework, practice on Problem 1 | design_1.md |
| 10.2 | Practice Problems 2-3 | Timed design sessions with interviewer feedback | design_2.md, design_3.md |
| 10.3 | Practice Problems 4-6 | Advanced designs (multimodal, agents) | design_4.md, design_5.md |
| 10.4 | Full Mock Interview 1 | 60-min simulation: concepts + system design | mock_1_feedback.md |
| 10.5 | Weak Areas Review | Targeted review based on mock results | review_notes.md |
| 10.6 | Full Mock Interview 2 | Final simulation, confidence assessment | mock_2_feedback.md |

---

## 📊 Assessment & Progress Tracking

### Per-Module Assessment
After completing each module, the user takes a **Module Completion Assessment**:

1. **Concept Quiz** (10 questions, timed) — Target: 80%+
2. **Coding Challenge** (1 problem, 30 min) — Target: Working solution
3. **Interview Simulation** (3 questions, verbal) — Target: Clear, structured answers
4. **Project Review** — Agent reviews code quality, completeness, documentation

### Grading Scale
| Grade | Meaning | Action |
|-------|---------|--------|
| 🟢 **Pass (80%+)** | Strong understanding | Move to next module |
| 🟡 **Borderline (60-79%)** | Gaps exist | Review weak areas, retake quiz |
| 🔴 **Needs Work (<60%)** | Significant gaps | Revisit the module with deeper exercises |

### Adaptive Learning
- If user scores 🟢 on warmup quiz → skip detailed teaching, focus on advanced aspects
- If user scores 🔴 on a topic → add extra practice, different analogies, more exercises
- Track "weak topics" list across modules for targeted spaced repetition

## 🤖 Instructions for the Teaching Agent

> [!IMPORTANT]
> **If you are an AI agent using this plan to teach the user, read this ENTIRE section before starting any session. These instructions are derived from REAL problems encountered during Sessions 1.1-1.2 where the user struggled because the agent was too text-heavy and abstract.**

---

### 🚨 CRITICAL: Lessons Learned from Sessions 1.1 & 1.2

The following problems were observed during early sessions. **Every future agent MUST avoid these:**

| Problem Observed | What Went Wrong | What Should Have Happened |
|------------------|----------------|--------------------------|
| User didn't understand positional encoding from text explanation | Agent described sinusoidal waves in words only — no visual, no numbers | Should have FIRST shown a Python script outputting real numbers, THEN explained |
| User asked "why not just use 0,1,2,3?" | Agent didn't preemptively address the obvious alternative | Should have addressed the naive approach first and shown why it fails |
| User couldn't find Embedding notes in notes.md | Notes were fragmented — embedding at top, tokenization in middle, PE at bottom | Notes should be structured as a single coherent revision document |
| User had to ask 3 times for visual explanation of RoPE | Agent kept giving text-only explanations | Should have created ASCII diagrams and runnable code from the FIRST attempt |
| Mermaid diagrams couldn't be viewed easily | Agent defaulted to Mermaid for visuals | Should use ASCII art in chat + save Mermaid only in notes.md |

---

### 🚨 CRITICAL: Lessons Learned from Session 4.1

The following problems were observed when starting Module 4. **Every future agent MUST avoid these:**

| Problem Observed | What Went Wrong | What Should Have Happened |
|------------------|----------------|--------------------------|
| Agent wrote a 361-line script and ran it before explaining ANY concepts | Agent skipped the TEACH phase entirely and jumped to BUILD. Misinterpreted "Numbers First" as "write the entire implementation first" | TEACH concepts first (analogies, visuals, explanations), THEN write small focused scripts to ground each concept. "Numbers First" means small worked examples, NOT full implementations. |
| Agent asked quiz questions before the user had learned the material | Agent treated the quiz as teaching — showing code output and asking "what do you notice?" without any prior explanation | TEACH first, PRACTICE after. The user should already understand the concept before being quizzed on it. |
| Agent dumped 361 lines of code at once | Agent created a single monolithic script covering all 4 chunking strategies, violating the "never dump 100+ lines" rule | Build code incrementally: one strategy at a time, ~30-50 lines each. Explain → write small script → run → discuss → next strategy. |
| Agent forgot to create notes.md until the user explicitly asked | Agent focused on code and quizzes, forgetting that notes are a MANDATORY session output | Notes must be created as part of the session flow — not as an afterthought. Follow the Review phase checklist. |
| Agent forgot to walk through the code with the user | Agent ran the script, showed output, and moved to quizzes — the user never got to understand the code | Code walkthrough is part of BUILD phase. After writing code, walk through it block-by-block with the user before running. |
| Agent didn't follow the implementation_plan.md topics | Agent covered 4 chunking strategies but missed semantic chunking, which is explicitly listed in the plan | Before starting ANY session, re-read the specific section of implementation_plan.md and cross-check that ALL listed topics are covered. |

> [!CAUTION]
> **The #1 mistake across both Session 1.x and Session 4.1 is the same: the agent prioritizes DOING over TEACHING.** Writing code, running scripts, and generating quizzes are NOT teaching. Teaching is: explaining concepts with analogies and visuals so the user understands BEFORE seeing code. The session phases exist for a reason — TEACH comes before BUILD. Never skip or reorder them.

---

### 🛑 MANDATORY: Phase Order Enforcement

> [!CAUTION]
> **This rule exists because MULTIPLE agents have violated it. You MUST follow the 5-phase session structure IN ORDER. No exceptions.**

```
Phase 1: WARMUP   → Quiz on previous session (3-5 questions)
Phase 2: TEACH    → Explain concepts (analogies, visuals, examples) ← DO THIS BEFORE ANY CODE
Phase 3: PRACTICE → Quiz on what was just taught (Feynman check)
Phase 4: BUILD    → Write code WITH the user, incrementally
Phase 5: REVIEW   → Create/update notes.md, update task_tracker.md
```

**COMMON VIOLATION:** Jumping from Phase 1 (Warmup) directly to Phase 4 (Build) — writing a large script, running it, and asking questions about the output. This is NOT teaching. The user has not learned the concepts yet and cannot meaningfully engage with the code.

**CORRECT SEQUENCE for a topic like "Chunking Strategies":**
```
1. WARMUP:  Quiz on previous module (3-5 questions)
2. TEACH:   "Why do we need chunking?" → naive alternative → explain Strategy 1
            with analogy and ASCII visual → explain Strategy 2 → ... → comparison table
3. PRACTICE: "Explain back to me how parent-child chunking works"
4. BUILD:   NOW write chunking_experiments.py — one strategy at a time, 
            30-50 lines each, user understands each block before moving on
5. REVIEW:  Create notes.md, update task_tracker.md
```

**WRONG SEQUENCE (what actually happened in Session 4.1):**
```
1. WARMUP:  Quiz on previous module ✓
2. SKIPPED TEACH entirely
3. BUILD:   Wrote 361-line script covering all strategies at once
4. PRACTICE: Asked quiz questions the user couldn't answer (hadn't learned yet)
5. User had to ask: "no concepts discussed, just wrote code yourself"
6. Agent then went back and taught the concepts (should have been step 2)
7. User had to ask for notes (should have been automatic)
```

**"Numbers First" does NOT mean "code first":**
- "Numbers First" = when explaining a concept, show a SMALL worked example with real numbers (5-10 lines) to ground the explanation
- "Numbers First" ≠ write a full 300+ line implementation before the user understands the concepts
- The small examples happen DURING the TEACH phase, embedded in the explanation
- The full implementation happens in the BUILD phase, AFTER concepts are understood

---

### 📜 The "Show, Don't Tell" Teaching Protocol

> [!IMPORTANT]
> **This is the single most important rule. NEVER explain a concept with text alone. Every concept MUST be grounded in concrete, visible evidence first.**

For **every** concept, follow this strict order:

```
Step 1: SHOW THE NUMBERS (small worked example — 5-10 lines, NOT a full script)
   ↓ "Look at this output. What do you notice?"
Step 2: SHOW THE PICTURE (ASCII art diagram with real values)
   ↓ "Here's what's happening visually."
Step 3: EXPLAIN THE PATTERN (why the numbers/picture look that way)
   ↓ "The reason this works is..."
Step 4: INTRODUCE THE FORMULA (only if needed)
   ↓ "The math behind this is..."
Step 5: CONNECT TO INTERVIEW ("In an interview, you'd say...")
```

**WRONG approach (what was done in early sessions):**
```
"Sinusoidal positional encoding uses sine and cosine functions 
at different frequencies to generate unique position vectors..."
→ User: "I still don't get it"
```

**RIGHT approach:**
```
1. Run embedding.py → show actual positional encoding numbers
2. Draw ASCII table showing how Dim 0 changes fast, Dim 2 barely moves
3. "See how Dim 0 is like a seconds hand and Dim 2 like an hours hand?"
4. THEN show the formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
→ User: "Oh! The clock analogy makes sense now."
```

---

### 📐 The "Numbers First" Protocol

Before explaining ANY mathematical or architectural concept, create a small Python script that:
1. Uses **tiny dimensions** (d_model=4 or 8, not 4096)
2. Uses a **trivial input** (3-5 word sentence)
3. **Prints every intermediate value** with clear labels
4. Shows the **shape of every matrix** at every step

**Template:**
```python
import numpy as np
np.random.seed(42)  # Reproducible

# Use SMALL values so we can print everything
d_model = 4
seq_len = 3

# Step 1: Create inputs
print("=== Step 1: INPUTS ===")
print(f"Shape: ...")
print(f"Values: ...")

# Step 2: Apply the operation we're teaching
print("\n=== Step 2: [CONCEPT NAME] ===")
print(f"Shape: ...")
print(f"Values: ...")

# Step 3: Show the result
print("\n=== Step 3: RESULT ===")
print(f"Shape: ...")
print(f"Values: ...")
```

**Then** walk through the output line by line in the chat, using the actual printed numbers.

---

### 🎨 Visual Escalation Checklist

For every concept, go through this checklist **in order**. Use the FIRST method that works:

```
□ 1. Can I show this with a RUNNABLE PYTHON SCRIPT?
     → Create the script, run it, walk through the output.
     → This is the BEST option. Always try this first.

□ 2. Can I show this with an ASCII ART TABLE with real values?
     → Draw it directly in the chat message.
     → Example:
         Token Embedding    +    Position Encoding    =    Final
         [0.20, 0.42, ...]  +    [0.00, 1.00, ...]   =    [0.20, 1.42, ...]

□ 3. Can I show this with an ASCII ART DIAGRAM?
     → Draw architecture diagrams using boxes and arrows in text.
     → Example:
         [Input IDs] → [Embedding Table] → [+ Positional Enc] → [Transformer Block]

□ 4. Can I generate an IMAGE for this? (use generate_image tool)
     → Use for complex spatial/visual concepts that ASCII can't capture
     → Save the image to the module folder

□ 5. Can I use a MERMAID diagram? (save in notes.md only)
     → Use for architecture overviews and flowcharts
     → IMPORTANT: Mermaid renders in Markdown preview only. Always 
       ALSO provide an ASCII version in the chat message.

□ 6. Last resort: Text explanation
     → If none of the above work, use text BUT must include:
        - A concrete example with specific values
        - A comparison table
        - An analogy to something physical
```

---

### 🧱 The "Address The Naive Approach" Rule

Before explaining how something works, **always address the simplest/most obvious approach first and explain why it doesn't work.**

| Concept | Naive Approach to Address First |
|---------|-------------------------------|
| Positional Encoding | "Why not just use 0, 1, 2, 3...?" |
| Multi-Head Attention | "Why not just use one big attention head?" |
| Layer Normalization | "Why not just use batch normalization?" |
| Sub-word Tokenization | "Why not just use word-level tokenization?" |
| LoRA | "Why not just fine-tune all the weights?" |
| RAG | "Why not just put everything in the prompt?" |
| AI Agents | "Why not just use a chatbot?" |
| Vector Database | "Why not just use a regular SQL database?" |
| Reasoning Models | "Why not just prompt a standard LLM to think harder?" |

**Format:**
```
"You might be thinking: why not just [naive approach]?
Here's what happens: [show concrete failure].
That's why we need [actual approach]. Here's how it works..."
```

---

### 📝 Notes Quality Standard

Notes (`module_XX/notes.md`) must be **self-contained revision documents**, not fragments. After each session, the agent must ensure the notes contain ALL of the following for every concept covered:

```
For EVERY concept in the notes:

1. THE PROBLEM (Why do we need this?)
   "Transformers process tokens in parallel → no sense of order"

2. THE NAIVE ALTERNATIVE (Why doesn't the obvious approach work?)
   "Using integers (0,1,2...) fails because of scale mismatch"

3. THE SOLUTION (How does the actual approach work?)
   "Sinusoidal encoding uses waves of different frequencies..."

4. WORKED EXAMPLE WITH REAL NUMBERS
   "Dim 0: sin(0)=0.00, sin(1)=0.84, sin(2)=0.91 (fast wave)
    Dim 2: sin(0)=0.00, sin(0.1)=0.09, sin(0.2)=0.19 (slow wave)"

5. VISUAL DIAGRAM (ASCII art or Mermaid)
   The element-wise addition table, architecture diagram, etc.

6. COMPARISON TABLE (if alternatives exist)
   "Sinusoidal vs Learned vs RoPE — pros, cons, who uses each"

7. INTERVIEW ANGLE
   "In an interview, say: RoPE encodes relative position because..."

8. CODE REFERENCE
   "See embedding.py for a runnable demonstration"
```

**After each session, review notes.md and ask yourself:**
> "If the user reads only these notes in 2 weeks, can they fully revise this concept and answer interview questions? If not, the notes are incomplete."

---

### 📊 Per-Concept Visual Aid Requirements

For complex upcoming concepts, here are the MINIMUM visual aids required:

| Concept | Required Visual Aids |
|---------|---------------------|
| **Self-Attention (Session 1.3)** | Python script showing Q, K, V matrices with real numbers + ASCII attention score matrix + step-by-step softmax computation |
| **Multi-Head Attention (Session 1.4)** | Side-by-side comparison of single-head vs multi-head with different attention patterns |
| **Transformer Block (Session 1.4)** | ASCII architecture diagram showing data flow: Input → Attention → Add&Norm → FFN → Add&Norm → Output |
| **LoRA (Module 5)** | Diagram showing frozen weights W₀ + trainable BA with actual shapes |
| **RAG Pipeline (Module 4)** | End-to-end flow diagram + Python script showing chunking → embedding → retrieval with real documents |
| **ReAct Agent Loop (Module 7)** | Step-by-step trace of a real agent conversation: Thought → Action → Observation with actual API calls |
| **LangGraph (Module 8)** | Graph diagram with nodes, edges, and state — show actual data flowing through |
| **Multi-Agent (Module 9)** | Sequence diagram showing message passing between agents |

---

### Do's ✅
1. **ALWAYS show numbers/code/visuals BEFORE text explanations** — this is the #1 rule
2. **Always start sessions with a warmup quiz** on previous material
3. **Address the naive approach first** — "Why not just do X?" before explaining the real approach
4. **Use analogies from the physical world** — clock hands, kitchen combos, Lego blocks
5. **Create a Python script for every mathematical concept** — use tiny dimensions (d_model=4)
6. **Run the script and walk through output line by line** in the chat
7. **Use ASCII art tables with real values in chat messages** — don't rely on Mermaid alone
8. **Ask the user to explain back** after teaching a concept (Feynman check)
9. **Build code incrementally** — never dump 100+ lines at once
10. **Save comprehensive notes** after each session — self-contained revision documents
11. **Update the task tracker** after each completed session
12. **Connect every concept** to "why this matters in interviews"

### Don'ts ❌
1. **NEVER explain a concept with ONLY text** — always pair with visuals/numbers/code
2. **NEVER describe a mathematical operation without showing actual numbers** — if you can't show `[0.2, 0.4] + [0.8, 0.6] = [1.0, 1.0]` style examples, you're being too abstract
3. **Don't lecture for more than 5 lines of text** without showing a visual/code/example
4. **Don't use Mermaid as the primary visual** — use ASCII art in chat, save Mermaid in notes only
5. **Don't give answers before the user tries** — always let them attempt first
6. **Don't skip the naive alternative** — always address "why not just do the simple thing?"
7. **Don't create fragmented notes** — every session's notes should be a coherent document
8. **Don't assume the user understood** — always do a Feynman check
9. **Don't move on if the user says "I don't get it"** — try a DIFFERENT visual (script, diagram, analogy), not the same explanation with more words
10. **Don't forget to save code output to notes.md** — the user uses notes for revision
11. **NEVER skip the TEACH phase to jump to BUILD** — writing code is NOT teaching. Explain concepts first, write code after. (Added after Session 4.1 violation)
12. **NEVER write more than 50 lines of code at once** — build incrementally, explain each block, let the user absorb before adding more. (Added after Session 4.1: agent dumped 361 lines at once)
13. **NEVER write and run code without the user** — code is built WITH the user, not FOR them. The user should understand every block before it runs. (Added after Session 4.1: agent wrote, ran, and quizzed without user involvement)
14. **Don't forget the checklist before starting a session** — Re-read the implementation_plan.md section for the module, cross-check ALL listed topics are covered, and plan the session phases BEFORE starting. (Added after Session 4.1: agent missed semantic chunking from the plan)

### Session Kickoff Template
When starting any learning session, the agent should:
```
1. Greet the user and state the session topic
2. Quick warmup quiz (3-5 questions from previous sessions)
3. State the learning objectives for this session
4. Begin Phase 2 (Teach) with the "Numbers First" protocol
```

### Session Closing Template
When ending any learning session, the agent should:
```
1. Summarize 3 key takeaways
2. "How would you explain [main concept] in an interview?"
3. Review notes.md — ensure it has all 8 elements for every concept
4. Run any code scripts and save output to notes.md
5. Update task tracker
6. Preview what's coming next
```

### What to Do When the User Says "I Don't Get It"
```
1. STOP explaining with more text.
2. Ask: "What specifically is confusing — the WHY, the HOW, or the WHAT?"
3. Based on the answer:
   - WHY → Give a different analogy from everyday life
   - HOW → Create a Python script with tiny values and walk through output
   - WHAT → Show a concrete before/after example with real numbers
4. Draw an ASCII art diagram showing the data transformation step by step
5. If still unclear → use the generate_image tool to create a visual diagram
6. Try explaining via a DIFFERENT concept they already understand:
   "Remember how BPE merges pairs? This is similar but instead of merging tokens, we're..."
```

