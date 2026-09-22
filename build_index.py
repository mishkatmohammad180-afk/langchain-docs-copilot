"""One-time script to ingest the full LangChain docs corpus and build the persisted vector store.
Run this whenever the docs are updated, not on every app startup."""

from app.ingestion import ingest_all_docs
from app.retrieval import build_vector_store

print("Ingesting full LangChain docs corpus (this will take a few minutes)...")
docs = ingest_all_docs()  # no limit = pulls everything

print(f"\nBuilding and persisting vector store with {len(docs)} chunks...")
build_vector_store(docs)

print("\nDone. Vector store persisted to data/chroma_db/")
