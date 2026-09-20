"""
LLM Client — Pure Python OpenRouter Integration
==================================================

DESIGN DECISION: Why use the `openai` Python SDK instead of LangChain?

LangChain's ChatOpenAI wraps the openai SDK with:
  - Chain composition (.invoke(), .pipe())
  - Prompt template rendering
  - Output parsers

But for a RAG system, we don't need any of that:
  - We build our own prompts (we want full control over citation instructions)
  - We parse the output ourselves (extract sources, confidence)
  - We want to see exactly what goes to the LLM (debugging)

The `openai` Python SDK is:
  - Already installed (dependency of langchain-openai)
  - The standard client for ALL OpenAI-compatible APIs (OpenRouter, Together, Groq, Ollama)
  - Zero abstraction — you see the exact request/response
  - Production-ready (retries, streaming, async support)

For an interview: "We use the openai SDK directly for LLM calls. LangChain
adds abstraction we don't need, and when something breaks, we'd rather debug
our own code than 12 layers of LangChain internals."
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass

# Ensure Windows stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LLMResponse:
    """
    Raw response from the LLM.

    We wrap the API response in our own dataclass so that:
      1. The rest of our code never imports `openai` directly
      2. If we switch from OpenRouter to Ollama or Azure, only this file changes
      3. We can add fields (latency, token count) without changing callers
    """
    content: str            # the generated text
    model: str              # which model actually responded
    prompt_tokens: int      # tokens in the prompt (cost tracking)
    completion_tokens: int  # tokens in the response
    total_tokens: int       # total tokens used

    def __repr__(self):
        return (f"LLMResponse(model={self.model}, "
                f"tokens={self.total_tokens}, "
                f"chars={len(self.content)})")


class LLMClient:
    """
    OpenRouter LLM client using the openai Python SDK.

    HOW OPENROUTER WORKS:
      OpenRouter provides a unified API for 100+ LLMs (GPT-4, Claude,
      Gemma, Llama, Mistral, etc.) using the OpenAI-compatible format.

      The trick: just point the openai SDK's base_url to OpenRouter:
        base_url = "https://openrouter.ai/api/v1"

      Everything else (chat completions, streaming, tool calling)
      works identically to the OpenAI API.

    WHY FREE MODELS?
      OpenRouter offers several models with a ":free" suffix that have
      no per-token cost. Perfect for learning and demos. For production,
      you'd use paid models (GPT-4o, Claude 3.5 Sonnet) for better quality.
    """

    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize the LLM client.

        Args:
            model: Model name on OpenRouter (e.g. "google/gemma-4-27b-it:free")
            temperature: 0 = deterministic, 1 = creative. We use 0 for RAG
                         because we want consistent, factual answers.
        """
        from openai import OpenAI
        from config import LLM_MODEL, LLM_API_BASE, LLM_API_KEY, LLM_TEMPERATURE, LLM_HEADERS

        self._model = model or LLM_MODEL
        self._temperature = temperature if temperature is not None else LLM_TEMPERATURE

        if not LLM_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set!\n"
                "  1. Copy .env.example to .env\n"
                "  2. Add your key: OPENROUTER_API_KEY=sk-or-v1-...\n"
                "  3. Get a free key at https://openrouter.ai/keys"
            )

        # Initialize the openai client pointing to OpenRouter
        self._client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE,
            default_headers=LLM_HEADERS,
        )

        print(f"  LLM Client initialized: {self._model}")

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """
        Send a prompt to the LLM and get a response.

        This is the CORE method. Everything else builds on top of it.

        HOW THE API CALL WORKS:
          We send a "chat completion" request with messages:
            [
              {"role": "system", "content": "You are a helpful assistant..."},
              {"role": "user", "content": "What was Acme's revenue?"}
            ]

          The LLM returns:
            {
              "choices": [{"message": {"content": "Acme's revenue was..."}}],
              "usage": {"prompt_tokens": 150, "completion_tokens": 50}
            }

        Args:
            prompt: The user's question (with context if doing RAG)
            system_prompt: Instructions for the LLM's behavior

        Returns:
            LLMResponse with the generated text and token usage
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )

            # Extract the response
            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model or self._model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )

        except Exception as e:
            # In production, you'd have retry logic, fallback models, etc.
            raise RuntimeError(f"LLM call failed: {e}")


# =============================================================================
# CLI — Test the LLM client directly
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM CLIENT — Direct Test")
    print("=" * 60)

    client = LLMClient()

    # Simple test
    print("\n  Sending test prompt...")
    response = client.generate(
        prompt="What is 2 + 2? Answer in one word.",
        system_prompt="You are a helpful assistant. Be concise."
    )

    print(f"\n  Response: {response.content}")
    print(f"  Model: {response.model}")
    print(f"  Tokens: {response.total_tokens} "
          f"(prompt={response.prompt_tokens}, "
          f"completion={response.completion_tokens})")
    print("\n✅ LLM client working!")
