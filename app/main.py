from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval import load_vector_store, load_documents_from_store, build_hybrid_retriever
from app.generation import generate_answer

retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Loading vector store and building retriever...")
    vectorstore = load_vector_store()
    documents = load_documents_from_store(vectorstore)
    retriever = build_hybrid_retriever(documents, vectorstore=vectorstore)
    print(f"Ready. Indexed {len(documents)} chunks.")
    yield
    print("Shutting down.")


app = FastAPI(title="LangChain Docs Copilot", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    retrieved_docs = retriever.invoke(request.question)
    answer = generate_answer(request.question, retrieved_docs)
    sources = list({
        doc.metadata.get("source_url")
        for doc in retrieved_docs
        if doc.metadata.get("source_url")
    })
    return AskResponse(answer=answer, sources=sources)
