"""
Strategy Comparison — Compare Dense vs Sparse vs Hybrid
=========================================================

This script runs the evaluation across all 3 retrieval strategies
and generates a side-by-side comparison table.

This is the kind of analysis you'd present in:
  - A design review ("why we chose hybrid over dense-only")
  - An interview ("how I evaluated my RAG system")
  - A production readiness review ("here are our quality metrics")
"""

import sys
import json
from pathlib import Path

# Ensure Windows stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_comparison(max_questions: int = None):
    """Run evaluation across all strategies and compare."""
    from evaluation.ragas_eval import run_evaluation, EvalReport
    from config import SAMPLE_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.document_loader import load_documents
    from ingestion.sample_data import create_sample_documents
    from ingestion.chunking import chunk_recursive

    # Load test dataset
    test_file = Path(__file__).parent / "test_dataset.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

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

    # Run evaluation for each strategy
    # (chunks are passed so run_evaluation can build BM25 on the chain's internal engine)
    strategies = ["dense", "sparse", "hybrid"]
    reports = {}

    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"  EVALUATING: {strategy.upper()}")
        print(f"{'=' * 70}")
        reports[strategy] = run_evaluation(
            test_data,
            strategy=strategy,
            max_questions=max_questions,
            chunks=all_chunks,
        )


    # Print comparison table
    print(f"\n\n{'=' * 70}")
    print(f"  STRATEGY COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<22} {'Dense':>10} {'Sparse':>10} {'Hybrid':>10}  {'Winner':<10}")
    print(f"  {'─' * 65}")

    metrics = [
        ("Faithfulness", "avg_faithfulness"),
        ("Answer Relevance", "avg_answer_relevance"),
        ("Context Precision", "avg_context_precision"),
        ("Context Recall", "avg_context_recall"),
        ("Avg Latency (s)", "avg_latency"),
    ]

    for label, attr in metrics:
        values = {s: getattr(reports[s], attr) for s in strategies}

        if attr == "avg_latency":
            # Lower is better for latency
            winner = min(values, key=values.get)
            val_strs = {s: f"{v:.1f}s" for s, v in values.items()}
        else:
            winner = max(values, key=values.get)
            val_strs = {s: f"{v:.1%}" for s, v in values.items()}

        print(f"  {label:<22} {val_strs['dense']:>10} {val_strs['sparse']:>10} "
              f"{val_strs['hybrid']:>10}  {'<-- ' + winner:<10}")

    # Save comparison report
    comparison = {
        "strategies_compared": strategies,
        "num_questions": len(test_data[:max_questions] if max_questions else test_data),
        "results": {},
    }
    for strategy, report in reports.items():
        comparison["results"][strategy] = {
            "avg_faithfulness": report.avg_faithfulness,
            "avg_answer_relevance": report.avg_answer_relevance,
            "avg_context_precision": report.avg_context_precision,
            "avg_context_recall": report.avg_context_recall,
            "avg_latency": report.avg_latency,
        }

    output_file = Path(__file__).parent / "comparison_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Comparison saved to: {output_file}")
    print(f"\n✅ Strategy comparison complete!")

    return reports


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run on 5 questions only (faster)")
    args = parser.parse_args()

    max_q = 5 if args.quick else None
    run_comparison(max_questions=max_q)
