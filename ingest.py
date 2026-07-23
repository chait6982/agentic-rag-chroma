"""
ingest.py
---------
Ingests a PDF into a persistent Chroma vector store.

Pipeline: load PDF -> chunk (word-window with overlap) -> embed with
OpenAI text-embedding-3-small -> store in Chroma (persistent, on disk).

This replaces the v1 hand-rolled TF-IDF/NumPy store with a production
vector database. The chunking logic is deliberately kept explicit (not
hidden behind a framework loader) so the retrieval behaviour stays
inspectable.

Usage:
    python ingest.py --pdf attention.pdf
    python ingest.py --pdf attention.pdf --collection attention_paper --reset
"""

from __future__ import annotations

import argparse
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION = "attention_paper"
CHUNK_WORDS = 220          # ~ a solid paragraph of context per chunk
CHUNK_OVERLAP_WORDS = 40   # overlap so sentences straddling chunks survive


def get_openai_ef() -> embedding_functions.OpenAIEmbeddingFunction:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY — set it in your .env file")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )


def load_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...] for pages with real text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path} — is it a scanned PDF?"
        )
    return pages


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """
    Word-window chunking with overlap, tracking source page numbers so
    answers can cite where each chunk came from.
    """
    chunks = []
    for page_num, text in pages:
        words = text.split()
        step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if len(piece.split()) < 30:   # skip tiny tail fragments
                continue
            chunks.append({
                "text": piece,
                "page": page_num,
            })
    return chunks


def ingest(pdf_path: str, collection_name: str, reset: bool) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_openai_ef(),
        metadata={"hnsw:space": "cosine"},
    )

    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")

    # Batch the adds (each add() call embeds via the OpenAI API)
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"], "source": os.path.basename(pdf_path)} for c in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )
        print(f"  embedded + stored {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{collection_name}' now holds {collection.count()} chunks "
          f"(persisted under ./{CHROMA_PATH}/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Chroma")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection")
    args = parser.parse_args()
    ingest(args.pdf, args.collection, args.reset)
