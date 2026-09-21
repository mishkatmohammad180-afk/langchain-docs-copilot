import re
import requests
from typing import List, Dict, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import GITHUB_REPO, DOCS_PATH

GITHUB_API_BASE = "https://api.github.com"


def get_repo_tree(repo: str = GITHUB_REPO, path: str = DOCS_PATH) -> List[Dict]:
    """Recursively list all markdown files under `path` in the GitHub repo."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/git/trees/main?recursive=1"
    response = requests.get(url)
    response.raise_for_status()
    tree = response.json()["tree"]

    return [
        item for item in tree
        if item["path"].startswith(path)
        and item["path"].endswith((".md", ".mdx"))
        and item["type"] == "blob"
    ]


def fetch_file_content(repo: str, file_path: str) -> str:
    """Fetch raw content of a single file from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def split_respecting_code_fences(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks without ever breaking inside a ```code``` fence."""
    fence_pattern = re.compile(r"(```.*?```)", re.DOTALL)
    segments = fence_pattern.split(text)

    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    buffer = ""

    for segment in segments:
        if segment.startswith("```"):
            if buffer.strip():
                chunks.extend(base_splitter.split_text(buffer))
                buffer = ""
            chunks.append(segment)
        else:
            buffer += segment

    if buffer.strip():
        chunks.extend(base_splitter.split_text(buffer))

    return chunks


def chunk_document(content: str, source_path: str) -> List[Document]:
    """Split by markdown headers first, then by size within each section."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_chunks = header_splitter.split_text(content)

    documents = []
    for chunk in header_chunks:
        for sub in split_respecting_code_fences(chunk.page_content):
            metadata = dict(chunk.metadata)
            metadata["source"] = source_path
            metadata["source_url"] = f"https://github.com/{GITHUB_REPO}/blob/main/{source_path}"
            documents.append(Document(page_content=sub, metadata=metadata))

    return documents


def ingest_all_docs(limit: Optional[int] = None) -> List[Document]:
    """Fetch and chunk markdown files under the docs path. Use `limit` to test on a subset first."""
    files = get_repo_tree()
    if limit:
        files = files[:limit]
    print(f"Found {len(files)} markdown files to ingest.")

    all_documents = []
    for i, file_info in enumerate(files):
        path = file_info["path"]
        try:
            content = fetch_file_content(GITHUB_REPO, path)
            all_documents.extend(chunk_document(content, path))
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(files)} files...")
        except Exception as e:
            print(f"Skipped {path}: {e}")

    print(f"Ingestion complete: {len(all_documents)} chunks from {len(files)} files.")
    return all_documents


if __name__ == "__main__":
    docs = ingest_all_docs(limit=10)
    if docs:
        print("\nSample chunk:")
        print(docs[0].page_content[:300])
        print(docs[0].metadata)
