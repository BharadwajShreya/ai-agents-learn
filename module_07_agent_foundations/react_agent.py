"""
ReAct Agent — Built from Scratch (No Frameworks!)
==================================================
Module 7, Session 7.3

This is a ReAct (Reason + Act) agent built with raw Python
and the Google Gemini API. No LangChain, no LlamaIndex, 
no frameworks — just the pure agent loop.

Architecture:
  User Query → System Prompt → Agent Loop → Response
  
Agent Loop:
  1. Send messages to LLM
  2. Parse response (extract Thought / Action / Final Answer)
  3. If Final Answer → return it
  4. If Action → execute the tool, get Observation
  5. Append Observation to messages → loop back to step 1
"""

import os
import re
import json
from pathlib import Path


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

def load_env_file():
    """Load environment variables from .env file (no extra dependency needed)."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env_file()

import google.generativeai as genai

# Read API key from environment (loaded from .env file)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
    raise ValueError(
        "❌ Set your API key in module_07_agent_foundations/.env file:\n"
        "   GEMINI_API_KEY=your-actual-key-here"
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

# Agent configuration
MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops


# ============================================================
# SECTION 2: TOOLS
# ============================================================
# Tools are just Python functions. The agent will call them
# by name with arguments. We start with ONE tool: calculator.

import urllib.request
import urllib.parse
import json

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Args:
        expression: A math expression like "15 * 3 + 7"
    
    Returns:
        The result as a string, or an error message
    """
    try:
        # Only allow safe math operations (no exec/eval of arbitrary code)
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Expression contains invalid characters. Only numbers and +-*/.() are allowed."
        
        result = eval(expression)  # Safe because we validated chars above
        return str(result)
    except Exception as e:
        return f"Error: Could not evaluate '{expression}'. {str(e)}"


def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return the summary of the top result."""
    try:
        # Clean the query for the URL
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
        
        # Wikipedia requires a descriptive User-Agent with contact info
        headers = {'User-Agent': 'ReActAgent/1.0 (https://github.com/agent; learning@example.com)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        search_results = data.get('query', {}).get('search', [])
        if not search_results:
            return f"No Wikipedia results found for '{query}'"
            
        # Get the title of the top result and fetch its summary
        top_title = search_results[0]['title']
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={urllib.parse.quote(top_title)}&format=json"
        
        req2 = urllib.request.Request(summary_url, headers=headers)
        with urllib.request.urlopen(req2) as response:
            summary_data = json.loads(response.read().decode())
            
        pages = summary_data.get('query', {}).get('pages', {})
        for page_id, page_info in pages.items():
            extract = page_info.get('extract', 'No summary available.')
            # Truncate if it's too long to save context window
            return f"Title: {top_title}\nSummary: {extract[:1000]}..."
            
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"


# Tool registry — maps tool names to functions
# This is how the agent knows which tools exist and how to call them
TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Evaluate a mathematical expression. Input: a math expression string like '15 * 3 + 7'",
    },
    "wikipedia": {
        "function": wikipedia_search,
        "description": "Search Wikipedia for facts, history, or people. Input: a search query string.",
    }
}


# ============================================================
# SECTION 3: SYSTEM PROMPT
# ============================================================
# This is the most important part! The system prompt teaches 
# the LLM HOW to be a ReAct agent. It defines the exact format
# for Thought / Action / Observation / Final Answer.

SYSTEM_PROMPT = """You are a ReAct (Reason + Act) agent. You solve problems by thinking step-by-step and using tools when needed.

## Available Tools:
{tool_descriptions}

## Response Format:
You MUST respond in EXACTLY one of these two formats:

### Format 1: When you need to use a tool
```
Thought: [Your reasoning about what to do next]
Action: tool_name(argument)
```

### Format 2: When you have the final answer
```
Thought: [Your reasoning about why you're done]
Final Answer: [Your complete answer to the user]
```

## Rules:
1. ALWAYS start with a Thought before any Action or Final Answer
2. Use EXACTLY ONE action per response — never multiple actions
3. After you receive an Observation (tool result), think about it before acting again
4. When you have enough information, give a Final Answer
5. Be precise with tool arguments — pass exactly what the tool expects
6. If a tool returns an error, think about why and try a different approach

## Example:
User: What is 25% of 180?

Thought: I need to calculate 25% of 180. That's 180 * 0.25. Let me use the calculator.
Action: calculator(180 * 0.25)

Observation: 45.0

Thought: The calculator returned 45.0. So 25% of 180 is 45. I have the answer.
Final Answer: 25% of 180 is 45.
"""


def build_system_prompt() -> str:
    """Build the system prompt with current tool descriptions."""
    tool_desc_lines = []
    for name, info in TOOLS.items():
        tool_desc_lines.append(f"- **{name}**: {info['description']}")
    tool_descriptions = "\n".join(tool_desc_lines)
    return SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)


# ============================================================
# SECTION 4: RESPONSE PARSER
# ============================================================
# The LLM returns free text. We need to parse out:
#   - The Thought (for debugging/transparency)
#   - The Action + arguments (to execute a tool)
#   - OR the Final Answer (to return to user)

def parse_response(response_text: str) -> dict:
    """Parse the LLM's response to extract Thought, Action, or Final Answer.
    
    Returns a dict with keys:
        - "thought": The agent's reasoning (always present)
        - "action": Tool name (if taking an action)
        - "action_input": Tool argument (if taking an action) 
        - "final_answer": The final response (if done)
    """
    result = {"thought": "", "action": None, "action_input": None, "final_answer": None}
    
    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", response_text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    
    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result
    
    # Check for Action
    action_match = re.search(r"Action:\s*(\w+)\((.+?)\)\s*$", response_text, re.MULTILINE)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_input"] = action_match.group(2).strip()
        # Remove surrounding quotes if present
        if result["action_input"].startswith('"') and result["action_input"].endswith('"'):
            result["action_input"] = result["action_input"][1:-1]
        elif result["action_input"].startswith("'") and result["action_input"].endswith("'"):
            result["action_input"] = result["action_input"][1:-1]
        return result
    
    # If we couldn't parse either, treat the whole response as a final answer
    # (graceful fallback — better than crashing)
    result["final_answer"] = response_text.strip()
    return result


# ============================================================
# SECTION 5: TOOL EXECUTOR
# ============================================================
# Takes a parsed action and runs the corresponding tool.

def execute_tool(action: str, action_input: str) -> str:
    """Execute a tool by name with the given input.
    
    Returns the tool's output as a string (the Observation).
    """
    if action not in TOOLS:
        return f"Error: Unknown tool '{action}'. Available tools: {list(TOOLS.keys())}"
    
    tool_fn = TOOLS[action]["function"]
    try:
        result = tool_fn(action_input)
        return result
    except Exception as e:
        return f"Error executing {action}: {str(e)}"


# ============================================================
# SECTION 6: THE AGENT LOOP — The Heart of the Agent!
# ============================================================
# This is the core ReAct loop:
#   Perceive → Reason (Thought) → Act (Action) → Observe → repeat

def run_agent(user_query: str, chat_history: list = None, verbose: bool = True) -> tuple:
    """Run the ReAct agent on a user query.
    
    Args:
        user_query: The user's question or task
        chat_history: List of dicts [{"role": "user/assistant", "content": "..."}]
        verbose: If True, print each step (Thought/Action/Observation)
    
    Returns:
        Tuple of (final_answer_string, updated_chat_history)
    """
    if chat_history is None:
        chat_history = []
        
    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 AGENT STARTED")
        print(f"📝 Query: {user_query}")
        print(f"{'='*60}")
    
    # Build the system prompt
    system_prompt = build_system_prompt()
    
    # Construct the full context string with memory!
    full_context = system_prompt + "\n\n=== CONVERSATION HISTORY ===\n"
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        full_context += f"{role}: {msg['content']}\n"
    
    full_context += f"\n=== CURRENT TASK ===\nUser: {user_query}\n"
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
        
        # STEP 1: Send to LLM (the agent's BRAIN reasons)
        response = model.generate_content(full_context)
        response_text = response.text.strip()
        
        # STEP 2: Parse the response
        parsed = parse_response(response_text)
        
        if verbose:
            print(f"💭 Thought: {parsed['thought']}")
        
        # STEP 3: Check if we have a Final Answer
        if parsed["final_answer"]:
            if verbose:
                print(f"✅ Final Answer: {parsed['final_answer']}")
                print(f"\n{'='*60}")
                print(f"🏁 AGENT FINISHED in {iteration} iteration(s)")
                print(f"{'='*60}")
                
            # Update memory before returning
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": parsed["final_answer"]})
            
            return parsed["final_answer"], chat_history
        
        # STEP 4: Execute the action (tool call)
        if parsed["action"]:
            if verbose:
                print(f"🔧 Action: {parsed['action']}({parsed['action_input']})")
            
            observation = execute_tool(parsed["action"], parsed["action_input"])
            
            if verbose:
                print(f"👁️ Observation: {observation}")
            
            # STEP 5: Add the observation to context and loop back
            # This is the agent's WORKING MEMORY (scratchpad)
            full_context += f"\n\n{response_text}\n\nObservation: {observation}"
        else:
            if verbose:
                print(f"⚠️ Could not parse action from response. Raw response:")
                print(f"   {response_text[:200]}")
            full_context += f"\n\n{response_text}\n\nObservation: I could not understand your response. Please use the exact format: 'Action: tool_name(argument)' or 'Final Answer: your answer'"
    
    # Safety: if we hit max iterations
    if verbose:
        print(f"\n⚠️ Agent hit max iterations ({MAX_ITERATIONS}). Returning last response.")
        
    fallback_answer = f"Agent stopped after {MAX_ITERATIONS} iterations without a final answer."
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": fallback_answer})
    
    return fallback_answer, chat_history


# ============================================================
# SECTION 7: TEST IT!
# ============================================================

if __name__ == "__main__":
    print("\n" + "🧪 TEST 1: Tool Chaining (Wiki + Math)")
    result, _ = run_agent("In what year was Google founded? Multiply that year by 5.", verbose=True)
    
    print("\n" + "🧪 TEST 2: Multi-turn Conversation (MEMORY TEST)")
    print("User: Who is the founder of NVIDIA?")
    history = []
    
    # Turn 1
    answer, history = run_agent("Who is the founder of NVIDIA?", chat_history=history, verbose=True)
    
    # Turn 2: Relies on memory of Turn 1
    print("\nUser: How old is he?")
    answer, history = run_agent("How old is he?", chat_history=history, verbose=True)
    
    # Turn 3: Relies on memory again
    print("\nUser: What was his age when he founded the company?")
    answer, history = run_agent("What was his age when he founded the company?", chat_history=history, verbose=True)
