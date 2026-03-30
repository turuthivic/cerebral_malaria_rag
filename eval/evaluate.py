"""
Phase 4: Evaluate RAG pipeline quality.

Usage:
    python -m eval.evaluate

Scores each answer on:
    - Relevance: does the answer address the question?
    - Groundedness: does it mention retrieved sources?
    - Topic coverage: how many expected topics appear in the answer?

Output:
    eval/results.json — detailed results per question
    Terminal summary with scores
"""

import json
import time
from pathlib import Path

from src.rag import RAGPipeline

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/results.json")


def score_topic_coverage(answer: str, expected_topics: list[str]) -> dict:
    """Check how many expected topics appear in the answer."""
    answer_lower = answer.lower()
    found = []
    missing = []
    for topic in expected_topics:
        if topic.lower() in answer_lower:
            found.append(topic)
        else:
            missing.append(topic)
    coverage = len(found) / len(expected_topics) if expected_topics else 0
    return {
        "coverage": round(coverage, 2),
        "found": found,
        "missing": missing,
    }


def score_groundedness(answer: str, sources: list[dict]) -> dict:
    """Check if the answer references sources."""
    answer_lower = answer.lower()
    # Check for source citation patterns
    has_citations = any(
        pattern in answer_lower
        for pattern in ["source", "[pubmed]", "[pmc]", "[who]", "[cdc]", "according to"]
    )
    # Check if answer content overlaps with source titles
    source_mentions = sum(
        1 for s in sources
        if any(word.lower() in answer_lower for word in s["title"].split()[:3] if len(word) > 4)
    )
    return {
        "has_citations": has_citations,
        "source_overlap": source_mentions,
        "score": min(1.0, (0.5 if has_citations else 0) + (source_mentions * 0.1)),
    }


def score_relevance(answer: str, question: str) -> dict:
    """Basic relevance check — does the answer address key terms from the question."""
    question_words = set(
        w.lower() for w in question.split()
        if len(w) > 3 and w.lower() not in {"what", "how", "does", "which", "that", "this", "from", "with", "have", "been", "were", "are", "the"}
    )
    answer_lower = answer.lower()
    matched = sum(1 for w in question_words if w in answer_lower)
    score = matched / len(question_words) if question_words else 0
    return {
        "score": round(score, 2),
        "matched_terms": matched,
        "total_terms": len(question_words),
    }


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    print("=" * 60)
    print(f"Evaluating {len(questions)} questions")
    print("=" * 60)

    rag = RAGPipeline()
    results = []
    totals = {"relevance": 0, "groundedness": 0, "coverage": 0}

    for q in questions:
        print(f"\n[{q['id']:2d}] {q['question']}")
        start = time.time()
        result = rag.query(q["question"])
        elapsed = round(time.time() - start, 1)

        relevance = score_relevance(result["answer"], q["question"])
        groundedness = score_groundedness(result["answer"], result["sources"])
        coverage = score_topic_coverage(result["answer"], q["expected_topics"])

        totals["relevance"] += relevance["score"]
        totals["groundedness"] += groundedness["score"]
        totals["coverage"] += coverage["coverage"]

        print(f"     Relevance: {relevance['score']:.0%}  "
              f"Grounded: {groundedness['score']:.0%}  "
              f"Coverage: {coverage['coverage']:.0%}  "
              f"({elapsed}s)")
        if coverage["missing"]:
            print(f"     Missing: {', '.join(coverage['missing'])}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "answer": result["answer"],
            "sources": result["sources"],
            "scores": {
                "relevance": relevance,
                "groundedness": groundedness,
                "topic_coverage": coverage,
            },
            "elapsed_seconds": elapsed,
        })

    n = len(questions)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions:    {n}")
    print(f"  Relevance:    {totals['relevance']/n:.0%}")
    print(f"  Groundedness: {totals['groundedness']/n:.0%}")
    print(f"  Coverage:     {totals['coverage']/n:.0%}")
    print(f"  Avg time:     {sum(r['elapsed_seconds'] for r in results)/n:.1f}s")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"relevance": 0, "coverage": 0, "count": 0}
        categories[cat]["relevance"] += r["scores"]["relevance"]["score"]
        categories[cat]["coverage"] += r["scores"]["topic_coverage"]["coverage"]
        categories[cat]["count"] += 1

    print(f"\nBy category:")
    for cat, vals in sorted(categories.items()):
        c = vals["count"]
        print(f"  {cat:15s}  relevance: {vals['relevance']/c:.0%}  coverage: {vals['coverage']/c:.0%}  (n={c})")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
