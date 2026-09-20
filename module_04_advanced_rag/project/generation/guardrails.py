"""
Guardrails — Hallucination Detection for RAG
===============================================

DESIGN DECISION: Why guardrails?

Even with perfect retrieval, LLMs can hallucinate:
  - Invent numbers not in the context
  - Attribute statements to the wrong source
  - Confidently answer when the context doesn't contain the answer

In production RAG systems, guardrails are ESSENTIAL:
  - Financial data: a hallucinated number could cause bad investment decisions
  - Legal/Medical: hallucination could have legal liability
  - Customer-facing: wrong answers erode trust

OUR APPROACH: Source Verification Guardrail
  After the LLM generates an answer, we check:
    1. Does the answer cite sources? (citation check)
    2. Are the cited source numbers valid? (source validation)
    3. Do key claims in the answer appear in the source chunks? (grounding check)

  This is a LIGHTWEIGHT guardrail. Production systems use:
    - A separate LLM call to verify faithfulness (expensive but thorough)
    - NLI (Natural Language Inference) models for entailment checking
    - RAGAS faithfulness metric (we'll implement this in Phase 4)
"""

import re
import sys
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

from retrieval.vector_store import SearchResult


@dataclass
class GuardrailResult:
    """Result of running guardrail checks on a RAG response."""
    passed: bool                    # overall pass/fail
    citation_check: bool            # does the answer cite sources?
    source_validation: bool         # are cited sources valid?
    grounding_score: float          # 0-1: what fraction of key numbers found in context
    warnings: list[str]             # specific issues found
    details: dict                   # raw check details

    def display(self):
        """Pretty-print guardrail results."""
        status = "PASSED" if self.passed else "FAILED"
        print(f"\n  {'─' * 60}")
        print(f"  GUARDRAIL CHECK: {status}")
        print(f"  {'─' * 60}")
        print(f"    Citation check:     {'✅' if self.citation_check else '❌'}")
        print(f"    Source validation:   {'✅' if self.source_validation else '❌'}")
        print(f"    Grounding score:    {self.grounding_score:.0%}")

        if self.warnings:
            print(f"\n    ⚠️  Warnings:")
            for w in self.warnings:
                print(f"      - {w}")


def check_guardrails(
    answer: str,
    sources: list[SearchResult],
    query: str,
) -> GuardrailResult:
    """
    Run guardrail checks on a RAG response.

    Args:
        answer: The LLM-generated answer
        sources: The source chunks that were provided as context
        query: The original user question

    Returns:
        GuardrailResult with pass/fail and detailed check results
    """
    warnings = []
    details = {}

    # =========================================================================
    # Check 1: Citation Presence
    # =========================================================================
    # Does the answer contain [Source N] citations?
    cited_sources = re.findall(r'\[Source\s*(\d+)\]', answer)
    cited_numbers = [int(n) for n in cited_sources]
    citation_check = len(cited_numbers) > 0

    if not citation_check:
        # Check for alternative citation patterns
        alt_citations = re.findall(r'(?:according to|based on|from)\s+\w+', answer, re.I)
        if alt_citations:
            citation_check = True
            warnings.append("Answer uses informal citations instead of [Source N] format")
        else:
            warnings.append("Answer does not cite any sources")

    details["cited_sources"] = cited_numbers

    # =========================================================================
    # Check 2: Source Validation
    # =========================================================================
    # Are the cited source numbers valid (within range)?
    valid_range = set(range(1, len(sources) + 1))
    invalid_citations = [n for n in cited_numbers if n not in valid_range]
    source_validation = len(invalid_citations) == 0

    if invalid_citations:
        warnings.append(
            f"Answer cites non-existent sources: {invalid_citations} "
            f"(only {len(sources)} sources provided)"
        )

    details["invalid_citations"] = invalid_citations

    # =========================================================================
    # Check 3: Grounding — Key Numbers Verification
    # =========================================================================
    # Extract numbers from the answer and check if they appear in the context
    # This catches hallucinated statistics/dollar amounts
    answer_numbers = _extract_numbers(answer)
    context_text = " ".join(s.text for s in sources)
    context_numbers = _extract_numbers(context_text)

    if answer_numbers:
        grounded_count = sum(1 for n in answer_numbers if n in context_numbers)
        grounding_score = grounded_count / len(answer_numbers)

        ungrounded = [n for n in answer_numbers if n not in context_numbers]
        if ungrounded:
            warnings.append(
                f"Numbers in answer not found in sources: {ungrounded[:5]}"
            )
    else:
        # No numbers in answer — can't check grounding, assume OK
        grounding_score = 1.0

    details["answer_numbers"] = answer_numbers
    details["grounded_numbers"] = [n for n in answer_numbers if n in context_numbers]
    details["ungrounded_numbers"] = [n for n in answer_numbers if n not in context_numbers]

    # =========================================================================
    # Check 4: "I don't know" detection
    # =========================================================================
    # If the answer says it can't find info, that's actually GOOD (not hallucinating)
    abstention_phrases = [
        "cannot find", "not found", "don't have", "no information",
        "not mentioned", "not available", "insufficient information",
    ]
    is_abstention = any(phrase in answer.lower() for phrase in abstention_phrases)
    if is_abstention:
        details["abstention"] = True
        # Abstention is never a guardrail failure
        grounding_score = 1.0

    # =========================================================================
    # Overall Pass/Fail
    # =========================================================================
    # Pass if: has citations AND valid source numbers AND >50% numbers grounded
    passed = citation_check and source_validation and grounding_score >= 0.5

    # Abstention always passes (correctly saying "I don't know")
    if is_abstention:
        passed = True

    return GuardrailResult(
        passed=passed,
        citation_check=citation_check,
        source_validation=source_validation,
        grounding_score=grounding_score,
        warnings=warnings,
        details=details,
    )


def _extract_numbers(text: str) -> list[str]:
    """
    Extract significant numbers from text.

    We extract:
      - Dollar amounts: $4.2B, $180 million, $630M
      - Percentages: 15%, 28.6%, 180 bps
      - Plain numbers with context: 47 new contracts, 3200 patents

    We SKIP:
      - Single digits (1, 2, 3) — too common, not useful for grounding
      - Years (2024, 2025) — present in both question and context
    """
    # Dollar amounts (e.g., $4.2B, $180M, $630 million)
    dollar_pattern = r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M|K))?'
    dollars = re.findall(dollar_pattern, text, re.I)

    # Percentages (e.g., 15%, 28.6%, 180 bps)
    pct_pattern = r'\d+(?:\.\d+)?%'
    percentages = re.findall(pct_pattern, text)

    # Significant numbers (2+ digits, not years)
    num_pattern = r'\b(\d{2,}(?:\.\d+)?)\b'
    raw_numbers = re.findall(num_pattern, text)
    # Filter out years
    significant = [n for n in raw_numbers
                   if not (len(n) == 4 and n.startswith(('19', '20')))]

    # Combine and deduplicate
    all_numbers = list(set(dollars + percentages + significant))
    return all_numbers


# =============================================================================
# CLI — Test guardrails on sample responses
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  GUARDRAIL CHECKS — Test")
    print("=" * 60)

    # Fake sources for testing
    fake_sources = [
        SearchResult(text="Revenue was $4.2 billion in Q3 2025, up 15%.", metadata={"source": "acme.md"}, score=0.9),
        SearchResult(text="Cloud revenue reached $2.1 billion.", metadata={"source": "acme.md"}, score=0.8),
    ]

    # Test 1: Good answer with citations
    print("\n  Test 1: Well-cited answer")
    good_answer = "Acme's revenue was $4.2 billion [Source 1], with cloud revenue at $2.1 billion [Source 2], representing 15% growth."
    result = check_guardrails(good_answer, fake_sources, "What was revenue?")
    result.display()

    # Test 2: Hallucinated number
    print("\n  Test 2: Hallucinated number")
    bad_answer = "Acme's revenue was $4.2 billion [Source 1]. Their profit margin was 42% [Source 2]."
    result = check_guardrails(bad_answer, fake_sources, "What was revenue?")
    result.display()

    # Test 3: No citations
    print("\n  Test 3: No citations")
    no_cite = "Acme had revenue of $4.2 billion and cloud revenue of $2.1 billion."
    result = check_guardrails(no_cite, fake_sources, "What was revenue?")
    result.display()

    # Test 4: Correct abstention
    print("\n  Test 4: Correct abstention")
    abstain = "I cannot find information about employee count in the provided documents."
    result = check_guardrails(abstain, fake_sources, "How many employees?")
    result.display()

    print("\n✅ Guardrail tests complete!")
