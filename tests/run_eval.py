"""Evaluation harness for the LangChain Docs Copilot.

Scores each question on:
  - Retrieval relevance: did we retrieve a chunk whose source URL contains
    the expected keyword? (skipped for out-of-scope questions)
  - Refusal correctness: did the system answer when it should, and decline
    (using the exact standardized refusal sentence) when it shouldn't?
  - Code syntax validity: does every Python code block in the answer parse
    correctly? Checks SYNTAX only, not execution.
"""

import ast
import json
import re
import textwrap
import time
from datetime import datetime, timezone

from app.retrieval import get_full_retriever
from app.generation import generate_answer, REFUSAL_MESSAGE
from tests.eval_set import EVAL_QUESTIONS


def looks_like_refusal(answer: str) -> bool:
    return REFUSAL_MESSAGE.lower() in answer.lower()


def extract_python_blocks(answer: str):
    return re.findall(r"```python[ \t]*\n(.*?)```", answer, re.DOTALL)


def check_code_syntax(answer: str):
    blocks = extract_python_blocks(answer)
    if not blocks:
        return True, "no code blocks found"
    for block in blocks:
        try:
            ast.parse(textwrap.dedent(block))
        except SyntaxError as e:
            return False, f"syntax error: {e}"
    return True, f"{len(blocks)} block(s) valid"


def run_evaluation():
    print("Loading retriever...")
    retriever = get_full_retriever()

    results = []
    for item in EVAL_QUESTIONS:
        print(f"\n[{item['id']}] {item['question']}")
        start = time.time()

        retrieved = retriever.invoke(item["question"])
        answer = generate_answer(item["question"], retrieved)
        elapsed = time.time() - start

        refused = looks_like_refusal(answer)

        if item["expected_source_keyword"]:
            retrieval_hit = any(
                item["expected_source_keyword"].lower() in (doc.metadata.get("source_url") or "").lower()
                for doc in retrieved
            )
        else:
            retrieval_hit = None

        refusal_correct = (not refused) if item["expects_answer"] else refused

        if item["expects_code"] and not refused:
            code_valid, code_note = check_code_syntax(answer)
        else:
            code_valid, code_note = None, "not applicable"

        result = {
            "id": item["id"], "question": item["question"],
            "retrieval_hit": retrieval_hit, "refusal_correct": refusal_correct,
            "code_valid": code_valid, "code_note": code_note,
            "latency_seconds": round(elapsed, 2), "answer_preview": answer[:200],
        }
        results.append(result)
        print(f"  Retrieval hit: {retrieval_hit} | Refusal correct: {refusal_correct} | Code: {code_note} | {elapsed:.1f}s")

        time.sleep(2)

    applicable_retrieval = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_score = sum(r["retrieval_hit"] for r in applicable_retrieval) / len(applicable_retrieval) if applicable_retrieval else 0
    refusal_score = sum(r["refusal_correct"] for r in results) / len(results)
    applicable_code = [r for r in results if r["code_valid"] is not None]
    code_score = sum(r["code_valid"] for r in applicable_code) / len(applicable_code) if applicable_code else 0
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "retrieval_relevance_score": round(retrieval_score, 3),
        "refusal_correctness_score": round(refusal_score, 3),
        "code_syntax_validity_score": round(code_score, 3),
        "average_latency_seconds": round(avg_latency, 2),
        "results": results,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Retrieval relevance:   {retrieval_score:.0%}  ({len(applicable_retrieval)} questions)")
    print(f"Refusal correctness:   {refusal_score:.0%}  ({len(results)} questions)")
    print(f"Code syntax validity:  {code_score:.0%}  ({len(applicable_code)} questions)")
    print(f"Average latency:       {avg_latency:.1f}s")

    with open("tests/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull results saved to tests/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
