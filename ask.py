"""
ask.py
------
Router-Retriever RAG v2 — Chroma edition.

Ask a question; a two-stage Router decides which of three paths answers it:

    1. PDF        -> retrieve top-k chunks from the Chroma vector store,
                     answer grounded in them with page citations
    2. WEB        -> DuckDuckGo search for current/live questions
    3. DIRECT     -> plain LLM answer for general reasoning

Routing is two-stage: a fast keyword heuristic first (zero LLM calls on
obvious questions -> lower latency and cost), falling back to a cheap LLM
classification only when the heuristic can't decide.

Every question produces a structured trace record (route taken, heuristic
vs LLM routing, retrieved contexts, final answer) appended to
traces/trace_log.jsonl for full observability.

Usage:
    python ask.py "What is multi-head attention?"
    python ask.py "What's the latest news about OpenAI?"
    python ask.py "What is 17 * 23?"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from ingest import get_openai_ef, CHROMA_PATH, DEFAULT_COLLECTION

load_dotenv()

client = OpenAI()  # reads OPENAI_API_KEY from env
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
TOP_K = 4

# --- Router: stage 1, keyword heuristic ---------------------------------

PDF_KEYWORDS = {
    "attention", "transformer", "encoder", "decoder", "multi-head",
    "positional", "embedding", "self-attention", "bleu", "paper",
    "vaswani", "softmax", "feed-forward",
}
WEB_KEYWORDS = {
    "latest", "today", "current", "news", "recent", "this week",
    "price", "now", "2025", "2026",
}


def route_heuristic(question: str) -> str | None:
    q = question.lower()
    if any(k in q for k in PDF_KEYWORDS):
        return "pdf"
    if any(k in q for k in WEB_KEYWORDS):
        return "web"
    return None  # undecided -> stage 2


def route_llm(question: str) -> str:
    resp = client.chat.completions.create(
        model=ROUTER_MODEL,
        max_tokens=5,
        messages=[
            {"role": "system", "content": (
                "Classify the question into exactly one route. Reply with one "
                "word only:\n"
                "pdf - if it is about the 'Attention Is All You Need' paper, "
                "transformers, or attention mechanisms\n"
                "web - if it needs current/live information from the internet\n"
                "direct - anything else (general reasoning, maths, definitions)"
            )},
            {"role": "user", "content": question},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip().lower()
    return answer if answer in {"pdf", "web", "direct"} else "direct"


# --- The three paths ------------------------------------------------------

def answer_from_pdf(question: str) -> tuple[str, list[str]]:
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(
        name=DEFAULT_COLLECTION, embedding_function=get_openai_ef()
    )
    res = collection.query(query_texts=[question], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    contexts = [f"[p.{m['page']}] {d}" for d, m in zip(docs, metas)]
    context_block = "\n\n".join(contexts)

    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer using ONLY the provided context from the paper. Cite "
                "page numbers like (p.3). If the context doesn't contain the "
                "answer, say so plainly."
            )},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, contexts


def answer_from_web(question: str) -> tuple[str, list[str]]:
    try:
        from ddgs import DDGS          # current package name
    except ImportError:
        from duckduckgo_search import DDGS  # older name, fallback

    with DDGS() as ddgs:
        hits = list(ddgs.text(question, max_results=5))
    contexts = [f"{h['title']}: {h['body']} ({h['href']})" for h in hits]
    context_block = "\n\n".join(contexts)

    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer using the web search results below. Attribute claims "
                "to their sources by name/URL."
            )},
            {"role": "user", "content": f"Search results:\n{context_block}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content, contexts


def answer_direct(question: str) -> tuple[str, list[str]]:
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content, []


# --- Trace logging --------------------------------------------------------

def log_trace(record: dict) -> None:
    Path("traces").mkdir(exist_ok=True)
    with open("traces/trace_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Entry point -----------------------------------------------------------

def ask(question: str) -> str:
    heuristic_route = route_heuristic(question)
    if heuristic_route is not None:
        route, router_stage = heuristic_route, "heuristic"
    else:
        route, router_stage = route_llm(question), "llm_fallback"

    if route == "pdf":
        answer, contexts = answer_from_pdf(question)
    elif route == "web":
        answer, contexts = answer_from_web(question)
    else:
        answer, contexts = answer_direct(question)

    log_trace({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "route": route,
        "router_stage": router_stage,   # heuristic (0 LLM calls) or llm_fallback
        "contexts": contexts,
        "answer": answer,
        "answer_model": ANSWER_MODEL,
    })

    print(f"\n[route: {route} via {router_stage}]\n")
    print(answer)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask the Router-Retriever RAG a question")
    parser.add_argument("question", help="Your question (quote it)")
    args = parser.parse_args()
    ask(args.question)
