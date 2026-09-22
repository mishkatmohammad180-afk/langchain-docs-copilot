from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import GROQ_API_KEY, LLM_MODEL

REFUSAL_MESSAGE = "The provided documentation does not contain information on this topic."

SYSTEM_PROMPT = f"""You are a LangChain documentation assistant with ONE hard rule that \
overrides everything else: NEVER answer using your own general knowledge. You may ONLY \
use the exact information given to you in the Context section below.

Before answering, silently check: does the Context actually contain information relevant \
to this specific question? If the question is not about LangChain or software \
development, or the Context does not address it, you MUST respond with exactly this \
sentence and nothing else: "{REFUSAL_MESSAGE}" Do not list any sources in that case.

If the Context IS relevant, answer using ONLY what it says, following these rules:
- Give a direct, working code example when the question asks for implementation.
- After your answer, list the sources you used, each as its exact source URL.
- Do not invent function names, parameters, or imports that are not shown in the context.
- Do not supplement with anything from your own training data, even if you are confident \
it is correct."""

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
    from app.retrieval import get_full_retriever

    retriever = get_full_retriever()

    query = "How do I use a HuggingFace agent with LangChain?"
    print(f"\nQuery: {query}\n")

    retrieved = retriever.invoke(query)
    print(f"Retrieved {len(retrieved)} chunks. Generating answer...\n")

    answer = generate_answer(query, retrieved)
    print("=" * 60)
    print(answer)
    print("=" * 60)
