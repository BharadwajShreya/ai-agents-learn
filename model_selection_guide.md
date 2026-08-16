# 🤖 Model Selection Guide — Which Model for Which Activity

> Use the right model for the right task to balance **quality, speed, and cost**.

---

## Quick Reference

| Activity | Best Model | Why |
|----------|-----------|-----|
| **Deep concept learning** (new topics) | **Claude Opus 4.6** | Best at long, nuanced explanations with analogies. Strongest teaching voice. |
| **Code projects & debugging** | **Claude Sonnet 4.6** | Excellent coder, fast enough for iterative development, cheaper than Opus. |
| **Quick quizzes & flashcards** | **Gemini Flash 3.5 (Medium)** | Fast, cheap, good enough for quiz Q&A. No need for deep reasoning. |
| **Architecture diagrams & research** | **Gemini 3.1 Pro (High)** | Strong at structured output, great with Mermaid diagrams, good at web research. |
| **Mock interviews & system design** | **Claude Opus 4.6** | Best at role-playing an interviewer, deep reasoning for system design critique. |
| **Simple lookups & quick answers** | **Gemini Flash 3.5 (Low)** | Fastest, cheapest. Use for "what does X mean?" type questions. |

---

## Detailed Breakdown by Learning Phase

### Phase 1: WARMUP (Quiz Review)
**Use: Gemini Flash 3.5 (Medium)**
- Warmup quizzes are rapid-fire recall questions
- Doesn't need deep reasoning — just needs to check your answers
- Fast response = better flow

### Phase 2: TEACH (Concept Learning)
**Use: Claude Opus 4.6**
- This is where you need the **best explanations**
- Opus excels at: analogies, progressive disclosure, connecting ideas, patient teaching
- It follows complex instructions well (the 5-phase session structure)
- Worth the premium for deep learning sessions

**Alternative: Gemini 3.1 Pro (High)**
- Also excellent for teaching, especially for topics involving latest research
- Better at generating visual diagrams (Mermaid) in some cases
- Use when you want a different "teaching voice" or perspective

### Phase 3: PRACTICE (Quizzes & Exercises)
**Use: Gemini Flash 3.5 (Medium) or Claude Sonnet 4.6**
- MCQ generation and grading doesn't need frontier reasoning
- Sonnet is better for open-ended grading ("evaluate my explanation")
- Flash is fine for MCQ and true/false

### Phase 4: BUILD (Coding)
**Use: Claude Sonnet 4.6**
- Sweet spot of coding quality + speed + cost
- Excellent at incremental code building (skeleton → fill in)
- Great at code review and suggesting improvements
- Fast enough for the iterative "write → run → fix" loop

**Upgrade to Claude Opus 4.6 when:**
- Debugging a tricky architectural issue
- Designing complex multi-file projects (Module 7-9 agents)
- Need to understand WHY something isn't working (not just fix it)

### Phase 5: REVIEW (Summary & Interview Practice)
**Use: Claude Opus 4.6**
- Best at simulating a realistic interviewer
- Gives the most nuanced feedback on your answers
- Can push back and ask follow-ups naturally

---

## Module-by-Module Recommendations

| Module | Primary Model | Reasoning |
|--------|--------------|-----------|
| **1. LLM Internals** | Claude Opus 4.6 | Math-heavy, needs careful explanations of attention, transformers |
| **2. Reasoning & Multimodal** | Claude Opus 4.6 | New concepts, needs deep comparative explanations |
| **3. Advanced Prompting** | Claude Sonnet 4.6 | Hands-on prompting experiments, iterative |
| **4. Advanced RAG** | Gemini 3.1 Pro (High) | Good at research, architecture diagrams, RAG is well-documented |
| **5. Fine-Tuning** | Claude Sonnet 4.6 | Code-heavy (HuggingFace, training loops) |
| **6. SLMs & Edge** | Claude Sonnet 4.6 | Practical (Ollama setup, benchmarking), code-focused |
| **7. Agent Foundations** | Claude Opus 4.6 | Building from scratch — needs deep understanding |
| **8. Frameworks & SDKs** | Claude Sonnet 4.6 | Code-heavy (LangGraph, MCP, SDKs) |
| **9. Multi-Agent & Production** | Claude Opus 4.6 | Complex architecture, production patterns, system thinking |
| **10. System Design** | Claude Opus 4.6 | Mock interviews, system design critique needs frontier reasoning |

---

## Cost-Saving Tips

1. **Start each session with Flash** for warmup quizzes → switch to Opus/Sonnet for teaching
2. **Use Sonnet for coding** — it's 80% as good as Opus at coding, significantly cheaper
3. **Use Flash for "can you remind me what X means?"** quick lookups
4. **Save Opus for:** first encounter with new concepts, mock interviews, system design, debugging complex issues
5. **Use Gemini Pro for:** topics where you want to cross-reference with Google's latest research

---

## The 80/20 Rule

If you want to keep it simple:

> **Use Claude Sonnet 4.6 for 80% of your learning.**
> **Upgrade to Claude Opus 4.6 for deep concept sessions and mock interviews.**
> **Use Gemini Flash 3.5 for quick quizzes and lookups.**

This gives you the best balance of quality, speed, and cost.
