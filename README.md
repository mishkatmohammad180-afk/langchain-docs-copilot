# LangChain Docs Copilot

A retrieval-augmented assistant that answers questions about the LangChain Python docs, cites the pages it used, and refuses when the docs don't cover the question.

**Live demo:** [ADD URL] · **2-minute walkthrough:** [ADD VIDEO LINK]

## What it does

- Ingests the LangChain Python docs (`langchain-ai/docs`, `src/oss/python`): 361 files, about 9,000 chunks.
- Retrieves with a hybrid of BM25 keyword search and vector search, then generates a grounded answer with source links.
- Declines to answer when the question falls outside the docs, instead of guessing.
- Ships with an evaluation harness so changes can be measured, not just eyeballed.

## Results

Measured with the harness in `tests/` (eval set: [N] questions):

| Metric | Result |
|---|---|
| Retrieval relevance | 100% |
| Refusal correctness (out-of-scope questions) | 93% |
| Code syntax validity (generated Python parses) | 100% |
| Average latency per question | ~6.8 s |

## Architecture

[ADD DIAGRAM IMAGE]

```
LangChain docs (GitHub)
        │  ingestion: header- and code-fence-aware chunking
        ▼
  Chroma vector store  +  BM25 index
        │  hybrid retrieval
        ▼
  LLM (Groq, openai/gpt-oss-120b)  →  answer + cited sources
        ▲
  FastAPI  POST /ask
```

- **Chunking:** splits on markdown headers and keeps code fences intact, so code examples aren't cut in half.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector store:** Chroma, persisted to `data/chroma_db`.
- **Generation:** Groq-hosted `openai/gpt-oss-120b`.
- **API:** FastAPI. Interactive docs at `/docs`.

## Project structure

```
app/
  config.py       settings (models, paths, repo to ingest)
  ingestion.py    fetches and chunks the docs
  retrieval.py    builds the vector store, hybrid retrieval
  generation.py   prompt + LLM call, citations, refusals
  main.py         FastAPI app
build_index.py    one-time script: ingest docs and build the index
tests/
  eval_set.py     evaluation questions
  run_eval.py     runs the evaluation
  eval_results.json
Dockerfile
```

## Run it with Docker

The image builds the index during `docker build` (about 7 minutes the first time), so the container starts ready to answer.

```bash
docker build -t langchain-docs-copilot .

export GROQ_API_KEY=your_key_here
docker run -p 8000:8000 -e GROQ_API_KEY langchain-docs-copilot
```

Then open `http://localhost:8000/docs` and try `POST /ask`:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create an agent in LangChain?"}'
```

The response contains an `answer` and a list of `sources`.

## Run it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env
python build_index.py                     # one-time: builds data/chroma_db
uvicorn app.main:app --reload
```

## Run the evaluation

```bash
python tests/run_eval.py
```

Results are written to `tests/eval_results.json`.

## Known limitations

- **Loosely related sources can appear.** For "how do I create an agent", the answer was correct, but two of the three cited pages were integration pages rather than the core agent docs. Planned fix: a similarity-threshold gate and a re-ranking step so weak matches are dropped before generation.
- **Latency of about 7 s** per question is dominated by the hosted LLM call. Streaming the response would improve perceived speed.
- **Static index.** The docs are indexed once at build time; rerun `build_index.py` (or rebuild the image) to pick up doc updates.
- **Docs only.** It answers from the Python docs it was given, not from the wider LangChain ecosystem.

## Tech stack

Python, FastAPI, LangChain, Chroma, sentence-transformers, BM25, Groq, Docker.
