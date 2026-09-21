from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from app.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vector_store(documents: List[Document]) -> Chroma:
    """Embed documents and persist them to a Chroma vector store."""
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def load_vector_store() -> Chroma:
    """Load an already-persisted Chroma vector store from disk."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def build_hybrid_retriever(documents: List[Document], vectorstore: Optional[Chroma] = None, k: int = 5):
    """Combine BM25 (keyword) and Chroma (semantic) retrieval into one hybrid retriever."""
    if vectorstore is None:
        vectorstore = load_vector_store()

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )


if __name__ == "__main__":
    from app.ingestion import ingest_all_docs

    print("Ingesting sample docs...")
    docs = ingest_all_docs(limit=10)

    print(f"\nBuilding vector store with {len(docs)} chunks...")
    vectorstore = build_vector_store(docs)

    print("Building hybrid retriever...")
    retriever = build_hybrid_retriever(docs, vectorstore=vectorstore)

    query = "How do I use a HuggingFace agent with LangChain?"
    print(f"\nTest query: {query}\n")
    results = retriever.invoke(query)

    for i, doc in enumerate(results[:3]):
        print(f"--- Result {i+1} ---")
        print(doc.page_content[:200])
        print(f"Source: {doc.metadata.get('source_url')}")
        print()
