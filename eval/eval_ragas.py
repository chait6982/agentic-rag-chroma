"""
eval_ragas.py
-------------
RAGAS evaluation harness for the Router-Retriever RAG (PDF path).

For each question in questions.json it:
    1. retrieves contexts from Chroma and generates an answer (same code
       path as ask.py's PDF route),
    2. pairs them with the human-written reference answer,
then computes four standard RAG metrics with RAGAS:

    faithfulness       - is the answer supported by the retrieved contexts?
    answer_relevancy   - does the answer actually address the question?
    context_precision  - are the retrieved chunks relevant to the question?
    context_recall     - do the retrieved chunks cover the reference answer?

Real metrics from real runs only — this script has no canned numbers.
RAGAS uses an LLM as judge, so OPENAI_API_KEY must be set and a full run
costs a small number of API calls per question.

Usage (from repo root):
    python eval/eval_ragas.py
    python eval/eval_ragas.py --questions eval/questions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from repo root or from eval/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from ask import answer_from_pdf  # reuse the exact production retrieval path

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def strip_page_tag(context: str) -> str:
    """Contexts carry a '[p.N] ' prefix for citations; RAGAS wants clean text."""
    if context.startswith("[p.") and "] " in context:
        return context.split("] ", 1)[1]
    return context


def build_dataset(questions_path: str) -> EvaluationDataset:
    with open(questions_path, encoding="utf-8") as f:
        golden = json.load(f)

    samples = []
    for i, item in enumerate(golden, start=1):
        q = item["question"]
        print(f"[{i}/{len(golden)}] running RAG for: {q}")
        answer, contexts = answer_from_pdf(q)
        samples.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": [strip_page_tag(c) for c in contexts],
            "reference": item["reference"],
        })

    return EvaluationDataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS")
    parser.add_argument("--questions", default="eval/questions.json")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY — set it in your .env file")

    dataset = build_dataset(args.questions)

    print("\nScoring with RAGAS (LLM-as-judge)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAS results ===")
    print(result)

    df = result.to_pandas()
    out_path = "eval/ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-question breakdown saved to {out_path}")


if __name__ == "__main__":
    main()
