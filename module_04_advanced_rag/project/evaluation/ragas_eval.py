"""
RAG Evaluation — Measuring Retrieval & Generation Quality
============================================================

DESIGN DECISION: Why evaluate RAG systems?

"It works on my 4 test queries" is NOT engineering.
Evaluation tells you:
  - WHICH component is failing (retrieval vs generation?)
  - WHETHER a change actually improved things (or just shifted failures)
  - HOW your system compares to baselines

THE 4 KEY RAG METRICS (from the RAGAS paper):

1. FAITHFULNESS (Generation quality)
   "Does the answer ONLY contain facts from the retrieved context?"
   High = no hallucination. Low = the LLM is making stuff up.
   
   HOW WE MEASURE IT:
     - Extract key claims (numbers, facts) from the answer
     - Check if each claim appears in the source context
     - Score = (claims found in context) / (total claims)

2. ANSWER RELEVANCE (Generation quality)
   "Does the answer actually address the question?"
   High = on-topic. Low = the LLM went on a tangent.
   
   HOW WE MEASURE IT:
     - Compute semantic similarity between question and answer
     - Score = cosine_similarity(embed(question), embed(answer))

3. CONTEXT PRECISION (Retrieval quality)
   "Are the TOP-ranked retrieved chunks actually relevant?"
   High = good ranking. Low = irrelevant chunks ranked high.
   
   HOW WE MEASURE IT:
     - Check if retrieved chunks contain info from the ground truth
     - Weight by rank position (top chunks matter more)

4. CONTEXT RECALL (Retrieval quality)
   "Did we retrieve ALL the chunks needed to answer?"
   High = nothing missed. Low = key information wasn't retrieved.
   
   HOW WE MEASURE IT:
     - Check what fraction of ground truth claims appear in retrieved chunks
     - Score = (ground truth facts found in context) / (total ground truth facts)

ARCHITECTURE:
  We implement these metrics in PURE PYTHON first (for understanding),
  then optionally run the RAGAS library for comparison.
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

# Ensure Windows stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    question: str
    ground_truth: str
    generated_answer: str
    faithfulness: float       # 0-1: answer grounded in context?
    answer_relevance: float   # 0-1: answer addresses the question?
    context_precision: float  # 0-1: top chunks are relevant?
    context_recall: float     # 0-1: all needed info retrieved?
    retrieved_sources: list = field(default_factory=list)
    latency_seconds: float = 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation report across all questions."""
    results: list[EvalResult]
    strategy: str
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_latency: float = 0.0

    def compute_averages(self):
        """Calculate average metrics across all results."""
        n = len(self.results)
        if n == 0:
            return
        self.avg_faithfulness = sum(r.faithfulness for r in self.results) / n
        self.avg_answer_relevance = sum(r.answer_relevance for r in self.results) / n
        self.avg_context_precision = sum(r.context_precision for r in self.results) / n
        self.avg_context_recall = sum(r.context_recall for r in self.results) / n
        self.avg_latency = sum(r.latency_seconds for r in self.results) / n

    def display(self):
        """Pretty-print the evaluation report."""
        self.compute_averages()
        print(f"\n  {'=' * 60}")
        print(f"  EVALUATION REPORT — Strategy: {self.strategy}")
        print(f"  {'=' * 60}")
        print(f"  Questions evaluated: {len(self.results)}")
        print(f"  Average latency:     {self.avg_latency:.1f}s per query")
        print(f"\n  {'─' * 60}")
        print(f"  METRIC SUMMARY")
        print(f"  {'─' * 60}")
        print(f"  Faithfulness:        {self.avg_faithfulness:.1%}  "
              f"{'✅' if self.avg_faithfulness >= 0.7 else '⚠️'}")
        print(f"  Answer Relevance:    {self.avg_answer_relevance:.1%}  "
              f"{'✅' if self.avg_answer_relevance >= 0.7 else '⚠️'}")
        print(f"  Context Precision:   {self.avg_context_precision:.1%}  "
              f"{'✅' if self.avg_context_precision >= 0.7 else '⚠️'}")
        print(f"  Context Recall:      {self.avg_context_recall:.1%}  "
              f"{'✅' if self.avg_context_recall >= 0.7 else '⚠️'}")

        # Per-question breakdown
        print(f"\n  {'─' * 60}")
        print(f"  PER-QUESTION BREAKDOWN")
        print(f"  {'─' * 60}")
        print(f"  {'#':<3} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6}  Question")
        print(f"  {'─' * 60}")
        for i, r in enumerate(self.results, 1):
            q_short = r.question[:40] + "..." if len(r.question) > 40 else r.question
            print(f"  {i:<3} {r.faithfulness:>5.0%} {r.answer_relevance:>5.0%} "
                  f"{r.context_precision:>5.0%} {r.context_recall:>5.0%}  {q_short}")


# =============================================================================
# Metric Implementations (Pure Python)
# =============================================================================

def compute_faithfulness(answer: str, context_texts: list[str]) -> float:
    """
    Measure: Does the answer ONLY contain facts from the context?

    Our approach (lightweight, no LLM needed):
      1. Extract "key facts" from the answer (numbers, percentages, dollar amounts)
      2. Check what fraction of those facts appear in the context
      3. Score = matched_facts / total_facts

    The RAGAS library uses an LLM to decompose the answer into atomic
    claims and verify each one — more thorough but costs LLM tokens.
    Our approach is free and catches the most common hallucination:
    invented numbers.
    """
    import re

    # Extract key facts from the answer
    answer_facts = _extract_key_facts(answer)

    if not answer_facts:
        # No verifiable facts in answer — assume faithful
        return 1.0

    # Check how many facts appear in any of the context chunks
    all_context = " ".join(context_texts)
    matched = sum(1 for fact in answer_facts if fact.lower() in all_context.lower())

    return matched / len(answer_facts)


def compute_answer_relevance(question: str, answer: str,
                             embed_model=None) -> float:
    """
    Measure: Does the answer actually address the question?

    Our approach:
      Compute cosine similarity between the question and answer embeddings.
      If the answer is about the same topic as the question, similarity is high.
      If the answer went off-topic, similarity is low.

    Score range: 0.0 (completely off-topic) to 1.0 (perfectly on-topic)
    """
    import numpy as np

    if embed_model is None:
        from config import get_embedding_model
        embed_model = get_embedding_model()

    q_emb = embed_model.embed_query(question)
    a_emb = embed_model.embed_query(answer)

    # Cosine similarity
    q_vec = np.array(q_emb)
    a_vec = np.array(a_emb)
    similarity = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))

    # Normalize to 0-1 range (cosine similarity for text is usually 0.3-0.9)
    # We stretch it so 0.5 cosine → 0.0 score, 1.0 cosine → 1.0 score
    score = max(0.0, min(1.0, (similarity - 0.5) * 2.0))
    return score


def compute_context_precision(retrieved_texts: list[str],
                              ground_truth: str) -> float:
    """
    Measure: Are the TOP-ranked retrieved chunks actually relevant?

    Our approach:
      For each retrieved chunk (ranked by position), check if it
      contains ANY of the key facts from the ground truth.
      Weight by position: chunk at rank 1 matters more than rank 5.

    Formula (weighted precision):
      score = Σ (is_relevant_i × weight_i) / Σ weight_i
      where weight_i = 1 / rank_i  (rank 1 gets weight 1.0, rank 5 gets 0.2)
    """
    ground_facts = _extract_key_facts(ground_truth)

    if not ground_facts:
        return 1.0  # no facts to check against

    weighted_sum = 0.0
    weight_total = 0.0

    for rank, chunk_text in enumerate(retrieved_texts, start=1):
        weight = 1.0 / rank  # rank 1 → weight 1.0, rank 5 → weight 0.2
        weight_total += weight

        # Check if this chunk contains any ground truth facts
        chunk_lower = chunk_text.lower()
        has_relevant_fact = any(fact.lower() in chunk_lower for fact in ground_facts)

        if has_relevant_fact:
            weighted_sum += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def compute_context_recall(retrieved_texts: list[str],
                           ground_truth: str) -> float:
    """
    Measure: Did we retrieve ALL the chunks needed to answer?

    Our approach:
      1. Extract key facts from the ground truth answer
      2. Check what fraction of those facts appear in the retrieved chunks
      3. Score = facts_found_in_retrieval / total_ground_truth_facts
    """
    ground_facts = _extract_key_facts(ground_truth)

    if not ground_facts:
        return 1.0  # nothing to recall

    all_retrieved = " ".join(retrieved_texts).lower()
    recalled = sum(1 for fact in ground_facts if fact.lower() in all_retrieved)

    return recalled / len(ground_facts)


def _extract_key_facts(text: str) -> list[str]:
    """
    Extract verifiable facts from text.

    We focus on NUMBERS because they're:
      1. Easy to extract with regex
      2. Easy to verify (exact match)
      3. The most dangerous when hallucinated (wrong $$ amounts)

    Extracts: dollar amounts, percentages, specific quantities
    """
    import re

    facts = []

    # Dollar amounts: $4.2B, $180 million, $630M, $2.56 billion
    dollar_pattern = r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M|K))?'
    facts.extend(re.findall(dollar_pattern, text, re.I))

    # Percentages: 15%, 28.6%, 180 bps
    pct_pattern = r'\d+(?:\.\d+)?%'
    facts.extend(re.findall(pct_pattern, text))

    # Specific quantities with units: 47 new contracts, 3200 patents, 45 AI researchers
    qty_pattern = r'\b(\d{2,})\s+(?:new |AI |)(?:contracts|patents|researchers|customers|employees)'
    qty_matches = re.findall(qty_pattern, text, re.I)
    facts.extend(qty_matches)

    return list(set(facts))  # deduplicate


# =============================================================================
# Evaluation Runner
# =============================================================================

def run_evaluation(
    test_data: list[dict],
    strategy: str = "hybrid",
    max_questions: int = None,
    chunks: list = None,
) -> EvalReport:
    """
    Run the full evaluation pipeline.

    Args:
        test_data: List of dicts with 'question', 'ground_truth', 'source_documents'
        strategy: Retrieval strategy to evaluate ("hybrid", "dense", "sparse")
        max_questions: Limit number of questions (for quick testing)
        chunks: Pre-computed chunks to build the BM25 index from.
                Required because BM25 is in-memory and lost on restart.

    Returns:
        EvalReport with per-question and aggregate metrics
    """
    from generation.rag_chain import RAGChain
    from config import get_embedding_model

    # Initialize
    chain = RAGChain(retrieval_strategy=strategy)
    embed_model = get_embedding_model()

    # Build BM25 index on the chain's INTERNAL search engine.
    # ChromaDB persists to disk, but BM25 is in-memory only.
    # We must rebuild it on the chain's own HybridSearchEngine instance.
    if chunks:
        chain._retrieval._sparse.index_chunks(chunks)
        print(f"  BM25 index rebuilt: {chain._retrieval._sparse.count} chunks")

    questions = test_data[:max_questions] if max_questions else test_data
    results = []

    print(f"\n  Evaluating {len(questions)} questions with strategy: {strategy}")
    print(f"  {'─' * 50}")

    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(questions)}] {question[:60]}...")

        # Run RAG pipeline and time it
        start_time = time.time()
        try:
            response = chain.ask(question, top_k=5, strategy=strategy)
            latency = time.time() - start_time

            # Extract context texts from retrieved sources
            context_texts = [s.text for s in response.sources]

            # Compute all 4 metrics
            faithfulness = compute_faithfulness(response.answer, context_texts)
            answer_relevance = compute_answer_relevance(
                question, response.answer, embed_model
            )
            context_precision = compute_context_precision(context_texts, ground_truth)
            context_recall = compute_context_recall(context_texts, ground_truth)

            result = EvalResult(
                question=question,
                ground_truth=ground_truth,
                generated_answer=response.answer,
                faithfulness=faithfulness,
                answer_relevance=answer_relevance,
                context_precision=context_precision,
                context_recall=context_recall,
                retrieved_sources=[s.metadata.get("source", "?") for s in response.sources],
                latency_seconds=latency,
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            result = EvalResult(
                question=question,
                ground_truth=ground_truth,
                generated_answer=f"ERROR: {e}",
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                context_recall=0.0,
                latency_seconds=0.0,
            )

        results.append(result)
        print(f"    Faith={result.faithfulness:.0%} Relev={result.answer_relevance:.0%} "
              f"Prec={result.context_precision:.0%} Recall={result.context_recall:.0%} "
              f"({result.latency_seconds:.1f}s)")

    report = EvalReport(results=results, strategy=strategy)
    report.compute_averages()
    return report


# =============================================================================
# CLI — Run evaluation on test dataset
# =============================================================================
if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    print("=" * 70)
    print("  RAG EVALUATION PIPELINE")
    print("=" * 70)

    # Load test dataset
    test_file = Path(__file__).parent / "test_dataset.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"\n  Loaded {len(test_data)} test questions")

    # Ensure documents are indexed
    if not any(Path(SAMPLE_DOCS_DIR).glob("*")):
        create_sample_documents(SAMPLE_DOCS_DIR)

    docs = load_documents(SAMPLE_DOCS_DIR)
    all_chunks = []
    for doc in docs:
        chunks = chunk_recursive(
            doc.content,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            source_metadata={"source": doc.metadata["source"]},
        )
        all_chunks.extend(chunks)

    # Run evaluation (use --quick flag for 5 questions, default is all 30)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run on 5 questions only (faster)")
    parser.add_argument("--strategy", default="hybrid",
                        choices=["hybrid", "dense", "sparse"])
    args = parser.parse_args()

    max_q = 5 if args.quick else None

    # Pass chunks so run_evaluation can build BM25 on the chain's internal engine
    report = run_evaluation(
        test_data,
        strategy=args.strategy,
        max_questions=max_q,
        chunks=all_chunks,
    )
    report.display()

    # Save results to JSON
    output_file = Path(__file__).parent / f"eval_results_{args.strategy}.json"
    eval_output = {
        "strategy": report.strategy,
        "num_questions": len(report.results),
        "avg_faithfulness": report.avg_faithfulness,
        "avg_answer_relevance": report.avg_answer_relevance,
        "avg_context_precision": report.avg_context_precision,
        "avg_context_recall": report.avg_context_recall,
        "avg_latency": report.avg_latency,
        "per_question": [
            {
                "question": r.question,
                "faithfulness": r.faithfulness,
                "answer_relevance": r.answer_relevance,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "latency": r.latency_seconds,
            }
            for r in report.results
        ],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n  Results saved to: {output_file}")
    print("\n✅ Evaluation complete!")
