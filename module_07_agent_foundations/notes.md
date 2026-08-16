# Module 7: AI Agents — Foundations & Architecture

## Session 7.1: What is an AI Agent? The Agent Loop

---

## 1. What is an AI Agent?

### THE PROBLEM — Why do we need agents?
LLMs are powerful reasoners but they are **frozen in time** and **can only talk**. They can't look up live data, call APIs, execute code, or take actions in the real world. A vanilla LLM asked "What's NVIDIA's stock price?" will say "I don't have real-time data."

### THE NAIVE ALTERNATIVE — Why not just use a chatbot?
A chatbot is a **one-shot** system: question in → answer out. It can only use knowledge baked into its weights. It has:
- ❌ No tools (can't search, calculate, query databases)
- ❌ No memory across steps (forgets intermediate results)
- ❌ No planning (can't break complex goals into sub-tasks)
- ❌ No autonomy (can't decide to take actions on its own)

### THE SOLUTION — The AI Agent
An AI agent wraps the **same LLM** in an **autonomous loop** with **tools, memory, and planning**. It keeps running until its goal is achieved.

> **One-line definition:** "A chatbot responds. An agent acts. An agent is an LLM wrapped in a loop, with tools, memory, and a goal — it keeps going until the goal is achieved."

### WORKED EXAMPLE

**Chatbot (fails):**
```
User: "What's NVIDIA's stock price and weekly change?"
Chatbot: "As of my training data, NVIDIA was ~$850. I don't have real-time data."
← STUCK. Conversation ends.
```

**Agent (succeeds):**
```
Thought: I need current stock data. Let me use my stock_price tool.
Action: stock_price(ticker="NVDA")
Observation: Current price: $1,247.50

Thought: Now I need last week's price to calculate change.
Action: stock_price(ticker="NVDA", date="2026-07-19")
Observation: Price on July 19: $1,185.20

Thought: Calculate: ($1247.50 - $1185.20) / $1185.20 = +5.26%
Action: respond("NVIDIA is at $1,247.50, up 5.26% from last week.")
← DONE ✅. Same LLM, radically different behavior.
```

### VISUAL DIAGRAM — Chatbot vs Agent

```
CHATBOT:
  User ──→ [LLM] ──→ Response       (one-shot, no loop)

AGENT:
  User ──→ [LLM + Tools + Memory + Planning] ──→ Response
                ↑                    │
                │    ┌───────────┐   │
                └────│   LOOP    │←──┘    (keeps going until goal met)
                     └───────────┘
```

### COMPARISON TABLE

| Feature | Chatbot | Agent |
|---------|---------|-------|
| Architecture | One-shot (input → output) | Loop (perceive → reason → act → observe) |
| Tools | ❌ None | ✅ Search, APIs, code, databases |
| Memory | Only current conversation | Short-term + long-term + working |
| Planning | ❌ None | ✅ Task decomposition, replanning |
| Autonomy | Responds only | Acts on the world |
| Knowledge | Frozen at training time | Live — accesses real-time data via tools |

### INTERVIEW ANGLE
> "A chatbot is a one-shot system — question in, answer out. An AI agent wraps the same LLM in an autonomous loop: it PERCEIVES the task, REASONS about what to do, ACTS using tools (search, APIs, code execution), and OBSERVES the result. It repeats this loop until the goal is achieved. The key components are the LLM (brain), tools (hands), memory (context across steps), and planning (task decomposition). The critical difference: a chatbot can only RESPOND, an agent can ACT on the world."

---

## 2. The Agent Loop

### THE PROBLEM — How does an agent actually work step by step?
We said agents run in a "loop" — but what exactly happens in each iteration?

### THE SOLUTION — The Perceive → Reason → Act → Observe Loop

Every AI agent, regardless of framework, runs this core loop:

```
                    ┌──────────────────────┐
                    │      USER GOAL       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
             ┌────→│  1. PERCEIVE         │  ← Read user input or previous observation
             │     └──────────┬───────────┘
             │                │
             │                ▼
             │     ┌──────────────────────┐
             │     │  2. REASON / PLAN    │  ← LLM generates a "Thought"
             │     └──────────┬───────────┘
             │                │
             │                ▼
             │     ┌──────────────────────┐
             │     │  3. ACT             │  ← Call a tool OR give final answer
             │     └──────────┬───────────┘
             │                │
             │                ▼
             │     ┌──────────────────────┐
             │     │  4. OBSERVE          │  ← Process the tool's result
             │     └──────────┬───────────┘
             │                │
             │                ▼
             │     ┌──────────────────────┐
             │     │  5. GOAL MET?        │
             │     └──────┬───────┬───────┘
             │         NO │       │ YES
             └────────────┘       ▼
                           FINAL RESPONSE ✅
```

### KEY INSIGHT
The loop terminates when the agent decides its goal is met. This decision is also made by the LLM — it generates a "final answer" action instead of a tool call.

### INTERVIEW ANGLE
> "The agent loop is: Perceive → Reason → Act → Observe → repeat. The LLM acts as the reasoning engine that decides what action to take at each step. The loop terminates when the LLM determines the goal has been achieved and generates a final response instead of another tool call."

---

## 3. The Four Pillars of an Agent

Every agent has four components that make the loop work:

```
┌────────────────────────────────────────────────────────────────┐
│                        AI AGENT                                │
│                                                                │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│   │  🧠 LLM  │  │ 🔧 TOOLS │  │ 💾 MEMORY│  │ 📋 PLANNING  │ │
│   │ The brain│  │ The hands│  │The notepad│  │ The strategy │ │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

| Pillar | Analogy | What It Does | Without It... |
|--------|---------|-------------|---------------|
| **🧠 LLM** | The brain | Reasons, decides next action | No intelligence — just a script |
| **🔧 Tools** | The hands | Interacts with the real world (search, APIs, code) | Can only talk, can't *do* anything |
| **💾 Memory** | The notepad | Remembers context & past results | Forgets what it just did 2 steps ago |
| **📋 Planning** | The strategy | Breaks goals into sub-tasks, replans | Gets lost on complex problems |

### INTERVIEW ANGLE
> "The four pillars of an agent are: the LLM (brain that reasons), tools (hands that interact with the world), memory (short-term and long-term context), and planning (task decomposition and replanning). Remove any one and the agent degrades significantly."

---

## 4. Core Agent Patterns

The agent loop is the skeleton. The **pattern** is the *strategy* for how the agent reasons inside that loop.

### Pattern 1: ReAct (Reason + Act)

**How it works:** Interleave reasoning and action — think before each step.

```
Thought:      "I need to find the weather in NYC"
Action:       search_weather(city="NYC")
Observation:  "72°F, sunny"
Thought:      "Now I can answer"
Action:       respond("It's 72°F and sunny in NYC")
DONE ✅
```

**Key insight:** The Thought is generated as plain text BEFORE the action. This makes reasoning **transparent and debuggable**.

**Best for:** Tool-heavy tasks (search, API calls, data retrieval)
**Weakness:** Can lose track on very long tasks

### Pattern 2: Plan-and-Execute

**How it works:** Create a full plan FIRST, then execute steps one by one.

```
PLANNING PHASE:
  Step 1: Search for NVIDIA's current price
  Step 2: Search for last week's price
  Step 3: Calculate percentage change
  Step 4: Format and respond

EXECUTION PHASE:
  Execute Step 1 → ✅
  Execute Step 2 → ✅
  Execute Step 3 → ✅
  Execute Step 4 → ✅ DONE
```

**Best for:** Complex multi-step workflows where upfront planning helps
**Weakness:** Rigid — doesn't adapt well if intermediate results change the plan

### Pattern 3: Reflection / Reflexion

**How it works:** Generate → Critique → Revise → repeat until quality threshold met.

```
Generate:  Write first draft of code
Critique:  "Missing error handling, variable names unclear"
Revise:    Fix error handling, rename variables
Critique:  "Looks good!" → DONE ✅
```

**Best for:** Quality-critical output (code generation, writing, analysis)
**Weakness:** Slow — requires multiple LLM calls for one output

### COMPARISON TABLE

| | ReAct | Plan-and-Execute | Reflection |
|---|---|---|---|
| **Style** | Think-Act-Observe interleaved | Plan first, then execute | Generate-Critique-Revise loop |
| **Adaptive?** | ✅ Very — adjusts each step | ❌ Less — follows the plan | ✅ Yes — improves iteratively |
| **Best for** | Tool-heavy tasks (search, APIs) | Complex multi-step workflows | Quality-critical output (code, writing) |
| **Weakness** | Loses track on long tasks | Rigid, doesn't adapt mid-execution | Slow, multiple LLM calls |
| **Real Example** | ChatGPT with web browsing | Devin (AI coder) plans then codes | Cursor/Copilot code review loop |

### COMBINING PATTERNS — The Real-World Approach

In production, agents **combine** patterns:

```
BLOG WRITING AGENT (combined approach):

Plan-and-Execute (top level):
  ├── Step 1: Research → uses ReAct (dynamic tool calling)
  ├── Step 2: Outline → single LLM call
  ├── Step 3: Write draft → single LLM call  
  └── Step 4: Review → uses Reflection (critique & revise)
```

### INTERVIEW ANGLE
> "The three core patterns are ReAct (interleaved reasoning and action — best for tool use), Plan-and-Execute (upfront planning — best for complex workflows), and Reflection (self-critique loop — best for quality-critical output). In practice, production agents combine them: Plan-and-Execute at the top level, ReAct for tool-calling steps, and Reflection for final quality checks."

---

## 📋 Session 7.1 Summary

| # | Concept | Key Takeaway |
|---|---------|-------------|
| 1 | Agent vs Chatbot | Chatbot responds, agent acts. Agent = LLM + loop + tools + memory + planning |
| 2 | Agent Loop | Perceive → Reason → Act → Observe → repeat until goal met |
| 3 | Four Pillars | LLM (brain), Tools (hands), Memory (notepad), Planning (strategy) |
| 4 | ReAct | Think → Act → Observe, interleaved. Transparent reasoning. |
| 5 | Plan-and-Execute | Plan everything first, then execute. Good for complex tasks. |
| 6 | Reflection | Generate → Critique → Revise loop. Good for quality-critical output. |
| 7 | Combining Patterns | Production agents mix patterns: P&E top-level, ReAct for tools, Reflection for polish |

---

## Session 7.2: Agent Memory, Planning & Tool Integration

---

## 5. Agent Memory

### THE PROBLEM — Why do we need structured memory?
As an agent takes multiple steps (calling tools, gathering data), the conversation history grows rapidly. A 15-step agent run can consume 45,000+ tokens. Raw conversation history creates three problems:
1. **Context window fills up** — old information gets pushed out
2. **Important info gets buried** — step 2's result is lost by step 15
3. **Cross-session amnesia** — user comes back tomorrow and the agent remembers nothing
4. **No learning from experience** — agent can't recall "last time I did X, it worked"

### THE NAIVE ALTERNATIVE — "Why not just pass all messages in the context window?"
That works for short conversations, but fails because:
- Context windows have limits (even 128K tokens fills up on complex tasks)
- No persistence across sessions (all context is lost when the session ends)
- No way to retrieve relevant past experiences (no similarity search)
- Cost scales linearly with context size (every token costs money)

### THE SOLUTION — Four Types of Structured Memory

#### Type 1: Short-Term Memory
- **What it stores:** Current conversation messages
- **Implemented as:** Chat history array passed in the LLM context window
- **Lifespan:** Single session (gone when session ends)
- **Analogy:** Your ability to hold ~7 items in your head right now

#### Type 2: Working Memory
- **What it stores:** Intermediate results, current plan, sub-task progress
- **Implemented as:** Scratchpad variable, agent state dictionary, or system prompt injection
- **Lifespan:** Single task (cleared after task completes)
- **Analogy:** The mental scratchpad you use while solving a math problem — "carry the 3, multiply by 5..."

#### Type 3: Episodic Memory
- **What it stores:** Specific past experiences WITH context and timestamps
- **Implemented as:** Vector database of past interactions, retrieved by similarity search
- **Lifespan:** Across sessions (persistent)
- **Analogy:** Your diary — "Last Tuesday, when I searched NVIDIA, I used the stock_price tool and found $1,247"
- **Key detail:** Stores the EXPERIENCE (what happened, what tools were used, what worked/failed), not just facts

#### Type 4: Long-Term Memory
- **What it stores:** User preferences, learned facts, domain knowledge (no specific timestamps)
- **Implemented as:** Database, vector store, or knowledge graph
- **Lifespan:** Permanent (persists across all sessions)
- **Analogy:** Your notebook of facts — "Paris is the capital of France" — you know it, even if you haven't thought about it in years

### WORKED EXAMPLE — All 4 Memory Types in Action

```
═════════════════════════════════════════════════════════
  DAY 1: User asks "Research NVIDIA's financials"
═════════════════════════════════════════════════════════

SHORT-TERM:   [all messages in current conversation]
WORKING:      { plan: ["get revenue", "get profit", "compare YoY"],
                current_step: 2, findings: { revenue_2025: "$130B" } }
EPISODIC:     (empty — first time doing this)
LONG-TERM:    { user_prefers: "tables", user_role: "ML engineer" }

═════════════════════════════════════════════════════════
  DAY 2: User asks "How is NVIDIA doing now?"
═════════════════════════════════════════════════════════

SHORT-TERM:   [fresh — Day 1 messages are GONE]
WORKING:      [fresh — Day 1 scratchpad is GONE]
EPISODIC:     → RETRIEVES: "Last time, I found NVIDIA revenue was $130B.
               I used stock_price and sec_filings tools. User liked tables."
               → Agent BUILDS ON previous research!
LONG-TERM:    { user_prefers: "tables", user_role: "ML engineer" }
               → Still knows preferences
```

### KEY DISTINCTION — Episodic vs Long-Term (Interview Trap!)

```
EPISODIC:   "I remember the TIME I did X"    → specific event with context
            "Last Tuesday, when I searched NVIDIA..."
            
LONG-TERM:  "I KNOW that X is true"          → general fact, no timestamp
            "The user prefers tables over paragraphs"

EPISODIC  = your DIARY     (specific events, with timestamps and context)
LONG-TERM = your NOTEBOOK  (facts and preferences, no specific date)
```

### COMPARISON TABLE

| Memory Type | Stores | Implemented As | Lifespan | Analogy |
|-------------|--------|---------------|----------|---------|
| **Short-term** | Current conversation | Chat history in context window | Single session | Holding items in your head |
| **Working** | Intermediate results, plan state | Scratchpad dict, system prompt | Single task | Mental math scratchpad |
| **Episodic** | Past experiences with context | Vector DB, similarity retrieval | Across sessions | Your diary |
| **Long-term** | Facts, preferences, knowledge | Database, knowledge graph | Permanent | Your notebook of facts |

### INTERVIEW ANGLE
> "Agent memory has four types: Short-term (conversation history in the context window — lasts one session), Working memory (scratchpad for intermediate results — lasts one task), Episodic memory (past experiences stored in a vector DB and retrieved by similarity — helps the agent learn from past interactions), and Long-term memory (persistent facts and user preferences). The key distinction interviewers test: episodic memory stores specific timestamped experiences ('last time I searched X'), while long-term memory stores general facts ('the user prefers tables')."

---

## 6. Planning & Task Decomposition

### THE PROBLEM — Why do agents need planning?
Without planning, agents on complex tasks **wander** — they take random steps, forget sub-tasks, go in circles, and produce incomplete output. Planning gives the agent **structure and direction**.

### THE NAIVE ALTERNATIVE — "Why not just let the LLM figure it out step by step?"
Pure ReAct (think-act-observe) works for simple tasks, but on complex multi-step tasks:
- The agent forgets early requirements by step 15
- No systematic coverage — it might skip entire sub-tasks
- No way to track progress or know when it's done
- Prone to going in circles on hard problems

### THE SOLUTION — Explicit Planning with Three Strategies

#### Strategy 1: Upfront Planning
- **How:** Create a complete plan BEFORE executing any steps
- **Good for:** Well-defined tasks where you know all the steps
- **Bad for:** Tasks where you don't know what you'll discover
- **Example:** "Convert this CSV to JSON" — steps are clear upfront

#### Strategy 2: Dynamic Replanning
- **How:** Create an initial plan → Start executing → Replan when new info arrives
- **Good for:** Research tasks, uncertain outcomes, exploratory work
- **Bad for:** Simple tasks (overkill)
- **Example:** "Research AI agents trends" — you discover new leads as you search, so the plan evolves

#### Strategy 3: Hierarchical Planning
- **How:** High-level plan → Each step has its own sub-plan
- **Good for:** Complex multi-phase tasks
- **Bad for:** Simple single-step tasks
- **Example:**
  ```
  Goal: "Write research report"
  ├── Phase 1: Research (sub-plan: search A, search B, ...)
  ├── Phase 2: Outline (sub-plan: intro, body, conclusion)
  └── Phase 3: Write (sub-plan: draft, review, polish)
  ```

### WORKED EXAMPLE — Agent With vs Without Planning

```
WITHOUT PLANNING (wanders):
  Step 1: search("cloud providers")
  Step 5: search("AWS pricing")
  Step 10: search("Azure features")
  Step 15: ← FORGOT it needed market share data
  Step 20: ← Going in circles
  Result: Incomplete, disorganized

WITH PLANNING (systematic):
  PLAN:
    1. Identify top 5 providers
    2. For each: get pricing, features, market share
    3. Create comparison table
    4. Write summary
  
  Execute Step 1 → ✅
  Execute Step 2a (AWS pricing) → ✅
  Execute Step 2b (AWS features) → ✅
  ...systematic, nothing missed
  Result: Complete, well-organized
```

### COMPARISON TABLE

| Strategy | When to Use | Advantage | Disadvantage |
|----------|-------------|-----------|-------------|
| **Upfront** | Well-defined tasks | Simple, efficient | Rigid, can't adapt |
| **Dynamic** | Research, uncertain tasks | Adapts to discoveries | More LLM calls (replanning cost) |
| **Hierarchical** | Complex multi-phase work | Organized, scalable | Complex to implement |

### INTERVIEW ANGLE
> "Planning in agents uses task decomposition — breaking a complex goal into manageable sub-tasks. Three strategies: Upfront planning (full plan then execute — good for defined tasks), Dynamic replanning (plan → execute → replan when new info arrives — good for research), and Hierarchical planning (nested sub-plans — good for multi-phase projects). Dynamic replanning is most common in production because real-world tasks rarely go exactly as planned."

---

## 7. Tool Integration & Tool Chaining

### THE PROBLEM — Why do agents need tools?
LLMs are frozen at training time — they can't access live data, execute code, send emails, or interact with external systems. Tools give the agent **hands** to act on the world.

### THE NAIVE ALTERNATIVE — "Why not just train the LLM to know everything?"
- Training data becomes stale immediately (no real-time stock prices)
- LLMs can't do precise math reliably (hallucinate calculations)
- LLMs can't interact with external systems (can't send emails, query databases)
- Impractical to train on all possible private/proprietary data

### THE SOLUTION — Tool Integration

#### How Tools Work
Tools are functions described to the LLM as JSON Schema. The LLM reads the description, decides when to use it, and generates the correct arguments.

```json
{
  "name": "web_search",
  "description": "Search the web for current information",
  "parameters": {
    "query": { "type": "string", "description": "The search query" },
    "num_results": { "type": "integer", "default": 5 }
  }
}
```

The LLM **pattern-matches** the user's intent against tool descriptions to select the right tool.

#### Tool Selection — How the LLM Decides

```
Available tools: web_search, calculator, stock_price, send_email

User: "What's 15% of NVIDIA's stock price?"

LLM reasoning:
  "I need NVIDIA's price → stock_price tool
   Then 15% of that → calculator tool"

Step 1: stock_price(ticker="NVDA") → $1,247.50
Step 2: calculator("1247.50 * 0.15") → $187.125
Response: "15% of NVIDIA's $1,247.50 is $187.13"
```

**Key insight:** Good tool descriptions are critical — if the description is vague, the LLM picks the wrong tool.

#### Tool Chaining — Output of One → Input of Another

```
QUERY: "Find the CEO of the highest-priced S&P 500 company 
        and get their LinkedIn profile."

CHAIN:
  stock_screener("S&P500, sort by price") → "AAPL"
       ↓ output becomes input
  company_info("AAPL") → "CEO: Tim Cook"
       ↓ output becomes input  
  web_search("Tim Cook LinkedIn") → URL
       ↓
  FINAL RESPONSE

Each tool's OUTPUT becomes the next tool's INPUT.
The LLM orchestrates the chain by reasoning at each step.
```

### TOOL ERROR HANDLING

```
COMMON STRATEGIES:
  1. RETRY      — Same tool again (for transient errors like timeouts)
  2. FALLBACK   — Different tool for same info (search API → web scraping)
  3. ASK USER   — "I couldn't find X. Can you provide it?"
  4. SKIP       — Proceed with available info, note what's missing
  5. MAX LOOPS  — Hard limit (e.g., 10 iterations) to prevent infinite loops
```

### TOOL SECURITY — Principle of Least Privilege

| Dangerous Tool | Risk | Safeguard |
|---|---|---|
| `send_email()` | Spam, phishing | Human-in-the-loop confirmation, recipient whitelist, rate limits |
| `delete_file()` | Data loss | Sandbox, whitelist allowed directories |
| `execute_code()` | Malicious code | Isolated container, no network access |
| `database_query()` | DROP TABLE | Read-only access, write requires approval |
| `make_payment()` | Unauthorized spending | Hard spending limits, human approval |

### INTERVIEW ANGLE
> "Tools give agents the ability to interact with the real world — search, compute, access APIs, execute code. The LLM selects tools by pattern-matching user intent against JSON Schema tool descriptions. Tool chaining is when one tool's output becomes the next tool's input, orchestrated by the LLM's reasoning. For security, follow the principle of least privilege: minimum permissions, human-in-the-loop for high-risk actions (send, delete, pay), sandboxed code execution, and full audit logging."

---

## 📋 Session 7.2 Summary

| # | Concept | Key Takeaway |
|---|---------|-------------|
| 1 | Agent Memory | 4 types: short-term (session), working (task), episodic (past experiences), long-term (permanent facts) |
| 2 | Episodic vs Long-term | Episodic = diary (timestamped events). Long-term = notebook (general facts). |
| 3 | Planning | 3 strategies: upfront (rigid), dynamic (adaptive), hierarchical (nested). Dynamic most common in production. |
| 4 | Task Decomposition | Break complex goals into sub-tasks. Without it, agents wander and miss requirements. |
| 5 | Tool Integration | Tools described as JSON Schema. LLM pattern-matches intent to select tools. |
| 6 | Tool Chaining | Output of one tool → input of another. LLM orchestrates the chain. |
| 7 | Tool Security | Least privilege, human-in-the-loop for risky actions, sandbox code, audit logs. |

---

*Next session: 7.3 — Build a ReAct Agent from Scratch (no frameworks!)*

# Session 7.3: Build a ReAct Agent from Scratch

---

## 1. The ReAct Architecture (Code Breakdown)

### THE PROBLEM — Frameworks abstract too much
When using LangChain or LlamaIndex to build agents, it's easy to treat the agent as a "black box". In an interview, you must be able to explain exactly what happens inside the loop.

### THE SOLUTION — Build from scratch
A complete ReAct agent requires only 5 core components:

1. **The System Prompt:** This is the most critical part. It defines the exact format the LLM must use:
   - Must use `Thought:` before every action
   - Must output `Action: tool_name(args)` to use a tool
   - Must output `Final Answer: ...` when done
   - Dynamically injects the list of available tools

2. **The Tools Registry:** A simple dictionary mapping string names to Python functions.
   - Requires robust error handling (a tool crash shouldn't crash the agent)
   - Requires strict security (e.g., validating calculator inputs before `eval()`)

3. **The Response Parser:** Regex functions that read the LLM's raw text and extract the Thought, the Action, and the arguments.

4. **The Context Manager (Memory):**
   - **Short-term memory:** Injects past conversation turns (`chat_history`) into the prompt.
   - **Working memory (Scratchpad):** Appends each `Thought + Action + Observation` to the current prompt string, so the LLM remembers its reasoning path.

5. **The Agent Loop:**
   ```python
   for i in range(MAX_ITERATIONS):
       response = llm(full_context)
       parsed = parse(response)
       if parsed["final_answer"]: 
           return answer
       observation = execute_tool(parsed["action"])
       full_context += f"{response}\nObservation: {observation}"
   ```

### KEY INSIGHTS (Interview Prep)

- **Why append to context?** By appending `Observation: {result}` to the prompt and sending it back, the LLM reads its own past thoughts and the tool results, allowing it to reason about the next step.
- **How does it know to stop?** The loop only breaks when the LLM decides to generate a `Final Answer` instead of an `Action`.
- **How does memory work?** `chat_history` resolves pronouns ("How old is he?") because past turns are prepended to the system prompt.

### GUARDRAILS — Protecting the Agent

1. **Infinite Loops:** Always use a `MAX_ITERATIONS` limit. If the LLM gets stuck retrying a broken tool, the hard limit terminates the run gracefully.
2. **Tool Safety:** Never blindly execute LLM output. The calculator tool validates characters (`0-9+-*/()`) before running `eval()` to prevent injection attacks.
3. **Graceful Degradation:** The `execute_tool` function wraps calls in `try/except` and returns the error *as a string* to the LLM. This allows the LLM to read the error (e.g., "403 Forbidden") and decide to try a different query!

### CODE REFERENCE
See `react_agent.py` in the module folder for the complete runnable implementation.
