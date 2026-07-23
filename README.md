# Agentic RAG v2 — Router-Retriever with Chroma + RAGAS Evaluation

A Router-Retriever RAG system over the *Attention Is All You Need* paper,
upgraded from a hand-rolled TF-IDF/NumPy store (v1) to a production
**Chroma vector database** with OpenAI embeddings — and evaluated with
**RAGAS** (faithfulness, answer relevancy, context precision, context
recall) computed from real runs.

## How it works

```
question
   │
   ▼
Router (two-stage)
   ├── stage 1: keyword heuristic  — zero LLM calls on obvious questions
   └── stage 2: LLM fallback (gpt-4o-mini) — only when the heuristic can't decide
   │
   ├── pdf    → Chroma top-k retrieval → GPT-4o answer with page citations
   ├── web    → DuckDuckGo search      → GPT-4o answer with source attribution
   └── direct → GPT-4o answers directly
   │
   ▼
structured trace record → traces/trace_log.jsonl
```

The two-stage router is a cost/latency design: most questions route on the
keyword heuristic alone (no router LLM call at all); the LLM classifier
only runs on genuinely ambiguous questions.

## v1 → v2: what changed and why

| | v1 | v2 (this repo) |
|---|---|---|
| Vector store | Hand-rolled TF-IDF + cosine in NumPy | **Chroma** (persistent, HNSW, cosine) |
| Embeddings | Term-frequency (sparse) | **OpenAI text-embedding-3-small** (dense semantic) |
| Evaluation | Manual inspection | **RAGAS**: 4 metrics, LLM-as-judge, per-question CSV |

Building v1 from primitives taught how retrieval actually works; v2 is
what you'd run in production — semantic (not lexical) matching, persistence
across sessions, and a real evaluation harness instead of eyeballing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in OPENAI_API_KEY
```

Download the paper (or use any PDF):
[Attention Is All You Need — arXiv PDF](https://arxiv.org/pdf/1706.03762) — save as `attention.pdf` in the repo root.

## Usage

**1. Ingest the PDF into Chroma** (one-time, ~1,000 embedding tokens):

```bash
python ingest.py --pdf attention.pdf
```

**2. Ask questions** — watch the router pick different paths:

```bash
python ask.py "What is multi-head attention?"          # → pdf (heuristic)
python ask.py "What's the latest news about OpenAI?"   # → web (heuristic)
python ask.py "What is 17 * 23?"                       # → direct (LLM fallback)
```

Every question appends a full trace (route, router stage, contexts,
answer) to `traces/trace_log.jsonl`.

**3. Evaluate with RAGAS:**

```bash
python eval/eval_ragas.py
```

Runs the 5-question golden set (`eval/questions.json`) through the live
PDF pipeline and scores it on faithfulness, answer relevancy, context
precision and context recall. Aggregate scores print to console; the
per-question breakdown lands in `eval/ragas_results.csv`.

> Metrics are computed from real runs — there are no canned numbers in
> this repo. RAGAS uses an LLM as judge, so an eval run makes a small
> number of OpenAI calls per question.

## Notes

- API keys load from `.env` (gitignored) — never hardcoded.
- `chroma_db/` and `traces/` are local artifacts, also gitignored;
  rebuild with `ingest.py`.
- Version pins in `requirements.txt` matter: `ragas==0.2.15` with
  `langchain-community<0.4` — newer langchain-community removes a module
  ragas imports.
## Results from a real run

RAGAS scores over the 5-question golden set (`eval/questions.json`), scored
with GPT-4o as judge:

| Metric | Score |
|---|---|
| Faithfulness | 0.967 |
| Context precision | 0.917 |
| Context recall | 0.800 |
| Answer relevancy | 0.777 |

Faithfulness at 0.97 means almost every claim in the generated answers is
supported by retrieved context — the anti-hallucination signal. Context
precision at 0.92 says the Chroma retrieval is surfacing genuinely relevant
chunks.

The weakest metric is answer relevancy (0.78): answers sometimes include
more surrounding detail than the question strictly asked for. The obvious
next levers are tightening the answer prompt for concision, and raising
`TOP_K` above 4 to lift context recall.