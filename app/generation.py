from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """You are a LangChain documentation assistant. Answer the developer's \
question using ONLY the provided context from LangChain's official documentation.

Rules:
- Give a direct, working code example when the question asks for implementation.
- After your answer, list the sources you used, each as its exact source URL.
- If the context doesn't contain enough information to answer confidently, say so \
explicitly instead of guessing.
- Do not invent function names, parameters, or imports that are not shown in the context."""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    return _llm


def format_context(documents: List[Document]) -> str:
    """Turn retrieved chunks into a labeled context block the LLM can cite from."""
    blocks = []
    for i, doc in enumerate(documents):
        source_url = doc.metadata.get("source_url", "unknown")
        blocks.append(f"[Source {i+1}: {source_url}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, documents: List[Document]) -> str:
    """Generate an answer to `query` grounded in the retrieved `documents`."""
    context = format_context(documents)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain = prompt | get_llm()
    response = chain.invoke({"context": context, "question": query})
    return response.content


if __name__ == "__main__":
    from app.ingestion import ingest_all_docs
    from app.retrieval import build_vector_store, build_hybrid_retriever

    print("Ingesting sample docs...")
    docs = ingest_all_docs(limit=10)

    print("Building retriever...")
    vectorstore = build_vector_store(docs)
    retriever = build_hybrid_retriever(docs, vectorstore=vectorstore)

    query = "How do I use a HuggingFace agent with LangChain?"
    print(f"\nQuery: {query}\n")

    retrieved = retriever.invoke(query)
    print(f"Retrieved {len(retrieved)} chunks. Generating answer...\n")

    answer = generate_answer(query, retrieved)
    print("=" * 60)
    print(answer)
    print("=" * 60)
