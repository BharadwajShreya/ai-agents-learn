"""
Evaluation package — RAGAS-style metrics and strategy comparison.
"""
from evaluation.ragas_eval import (
    run_evaluation, EvalResult, EvalReport,
    compute_faithfulness, compute_answer_relevance,
    compute_context_precision, compute_context_recall,
)
