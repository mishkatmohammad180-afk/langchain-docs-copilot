# LangChain Docs Copilot

A retrieval-augmented generation (RAG) assistant that answers implementation
questions about LangChain using LangChain's own official documentation —
with hybrid search, grounded/cited answers, and an automated evaluation harness.

Built as a portfolio project to demonstrate production-oriented RAG
engineering: not just "retrieve and generate," but chunking that respects
document structure, hybrid retrieval, a hallucination guardrail, and a real
evaluation suite that measures how well the guardrail actually holds up.

## Features

- **GitHub-based ingestion pipeline** — pulls markdown docs directly from
  `langchain-ai/docs`, chunks by markdown header first (keeping sections
  coherent) and never splits inside a code fence.
- **Hybrid retrieval** — combines BM25 (keyword matching, strong on exact
  API/function names) with Chroma semantic vector search via an
  `EnsembleRetriever`, using free local embeddings (`sentence-transformers`).
- **Grounded generation** — Groq-hosted LLM (`openai/gpt-oss-120b`), with a
  system prompt that requires citing real source URLs and refuses to answer
  when the retrieved context doesn't cover the question, rather than
  inventing plausible-looking but fake API details.
- **FastAPI backend** — `/ask` and `/health` endpoints, with the expensive
  retriever setup done once at startup rather than per-request.
- **Evaluation harness** — 15 hand-written questions (10 answerable, 5
  deliberately out-of-scope) scoring retrieval relevance, refusal
  correctness, and code syntax validity.

## Evaluation Results

| Metric | Score |
|---|---|
| Retrieval relevance | 100% (10/10) |
| Refusal correctness | 93% (14/15) |
| Code syntax validity | 100% (10/10) |
| Average latency | ~6.8s |

Full per-question results in `tests/eval_results.json`.

## Known Limitations

- **Refusal isn't perfect on high-confidence general knowledge.** The one
  failing eval case asks a trivial general-knowledge question ("What is the
  capital of France?"). The model answers from its own training instead of
  declining, even with an explicit, first-position system prompt instruction
  not to. This is a documented RAG failure mode (parametric vs. contextual
  knowledge conflict) — prompting alone has a real ceiling here. The correct
  fix is a **similarity-score gate**: check the top retrieved chunk's
  relevance score before generation, and skip the LLM call entirely if
  nothing relevant was found. Not yet implemented; a clear next step.
- **Code validation checks syntax, not execution.** Verifying generated code
  actually *runs* would mean executing arbitrary LLM output against real API
  keys for a dozen different services — not practical or safe for this
  project. `ast.parse` catches malformed code but not runtime errors.
- **`langchain_community` is being deprecated upstream** (per LangChain's own
  migration notice) — `BM25Retriever` currently lives there; a future
  update should migrate to its standalone package once one stabilizes.
- **Auto-refresh ingestion** (re-indexing when LangChain's docs repo gets new
  commits) is designed but not yet automated — currently a manual
  `python build_index.py` run.

## Setup

\`\`\`bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
\`\`\`

Create a `.env` file with:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`

Build the index (one-time, ~5-10 minutes):
\`\`\`bash
python build_index.py
\`\`\`

Run the API:
\`\`\`bash
uvicorn app.main:app --reload
\`\`\`

Interactive docs at `http://127.0.0.1:8000/docs`.

Run the evaluation suite:
\`\`\`bash
python -m tests.run_eval
\`\`\`

## Tech Stack

Python, FastAPI, LangChain, Chroma, Groq (Llama/GPT-OSS via free tier),
sentence-transformers, BM25.
