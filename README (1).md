# 🔎 NLP Search System

A semantic search system built on ML research papers that goes beyond keyword matching. It searches by **meaning**, then wraps every result with a **summary, keywords, a difficulty rating, and related searches** — turning a simple similarity lookup into a full mini search engine.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Example Output](#example-output)
- [Function Reference](#function-reference)
- [Notes & Limitations](#notes--limitations)
- [Possible Improvements](#possible-improvements)
- [Acknowledgements](#acknowledgements)

---

## Overview

Most search tools match exact words. This project matches **meaning**. A query like `"training neural networks faster"` can surface a paper that never uses the word "faster" at all, because the system compares the *embedding* of the query against the *embeddings* of every paper abstract, not the raw text.

On top of that core search, the system layers four NLP features so a single query returns a complete, useful package instead of just a list of titles:

1. A short **summary** of the best matching paper
2. The paper's key **keywords**
3. A judgment of how **difficult** the query itself is
4. A handful of **related searches** to explore next

## Features

| Feature | What it does |
|---|---|
| **Semantic Search** | Finds the most relevant papers using sentence embeddings + FAISS, so results are ranked by meaning rather than word overlap |
| **Summarization** | Condenses a long, dense abstract into a short, readable summary |
| **Keyword Extraction** | Surfaces the most representative terms from the top result |
| **Difficulty Level** | Classifies the query itself as Beginner, Intermediate, or Advanced |
| **Related Searches** | Suggests related topics pulled from the actual top search results (not a static list) |

## Architecture

```
                     ┌─────────────┐
   User Query ──────▶│  Embedding  │  (Sentence-Transformers)
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ FAISS Index │  ── top-k similar papers
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────────┬───────────────────┐
              ▼             ▼                 ▼                   ▼
        ┌──────────┐  ┌───────────┐   ┌───────────────┐   ┌──────────────┐
        │ Summarize│  │ Extract   │   │  Classify      │   │  Related     │
        │ (BART)   │  │ Keywords  │   │  Difficulty    │   │  Searches    │
        │          │  │ (KeyBERT) │   │  (Zero-Shot)   │   │  (KeyBERT on │
        │          │  │           │   │                │   │  top-k docs) │
        └──────────┘  └───────────┘   └───────────────┘   └──────────────┘
              │             │                 │                   │
              └─────────────┴─────────────────┴───────────────────┘
                                     │
                                     ▼
                           Final combined output
```

## Tech Stack

| Component | Library / Model |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Search | `faiss` (cosine similarity via inner product on normalized vectors) |
| Summarization | `transformers` (`facebook/bart-large-cnn`) |
| Keyword Extraction | `KeyBERT` |
| Difficulty Classification | `transformers` zero-shot classification (`facebook/bart-large-mnli`) |
| Data Handling | `pandas` |
| Dataset Loading | Hugging Face `datasets` |

## Dataset

- **Source:** `CShorten/ML-ArXiv-Papers` (via Hugging Face `datasets`)
- **Content:** Titles and abstracts of machine learning papers from arXiv
- **Used for:** building the searchable corpus — every abstract is embedded once and stored in a FAISS index for fast lookup

## How It Works

1. **Search** — the query is embedded and compared against a FAISS index of paper embeddings to retrieve the top-k matches.
2. **Summarize** — the top matching abstract is passed through a BART summarization model to produce a short summary.
3. **Extract Keywords** — KeyBERT pulls the most relevant terms out of that same abstract.
4. **Classify Difficulty** — a zero-shot classification model scores the *query itself* against the labels `beginner`, `intermediate`, and `advanced`, and the top-scoring label is returned along with each label's confidence score.
5. **Suggest Related Searches** — KeyBERT runs again, this time across the other top-k search results (not just the top one), and the extracted terms are combined into a short list of related searches.

All five steps are wrapped into a single function, `full_search_query()`, so a user only ever has to call one thing.

## Project Structure

```
NLPSearchSystem1.ipynb   # the full notebook: data loading, indexing, search, and all NLP features
README.md                # this file
```

## Setup

```bash
pip install datasets pandas sentence-transformers scikit-learn faiss-cpu
pip install transformers==4.46.3
pip install keybert==0.8.5
```

> Recommended: run in Google Colab with a GPU runtime (T4 or better) — the summarization and difficulty models are large and much faster on GPU.

## Usage

Run the notebook cells in order (data loading → embeddings → FAISS index → summarizer → KeyBERT → difficulty classifier), then call:

```python
full_search_query("your query here", k=5)
```

- `query` — the search text
- `k` — how many papers to retrieve from the index (default 5). More results feed into a richer "Related Searches" list.

## Example Output

```
>>> full_search_query("medical research python fastapi")

Summary
This paper presents a deep learning approach for medical image analysis,
demonstrating improved accuracy over traditional methods on clinical datasets.

Keywords
- medical imaging
- deep learning
- clinical data

Difficulty
Intermediate

Reason
- Intermediate score: 0.61
- Advanced score: 0.25
- Beginner score: 0.14

Related Searches
- clinical data
- healthcare
- bioinformatics
- neural networks
- diagnosis
- classification
```

## Function Reference

| Function | Purpose |
|---|---|
| `full_search_query(query, k=5)` | Runs the entire pipeline end-to-end and prints the combined result |
| `check_difficulty(query)` | Returns `(level, reasons)` — the difficulty label and per-label confidence scores |
| `get_related_searches(result_indexes)` | Takes FAISS result indices and returns a list of related keywords pulled from those papers |

## Notes & Limitations

- `facebook/bart-large-mnli` and `facebook/bart-large-cnn` are large models (~1.6GB each) — the first run downloads them, so expect a delay.
- The corpus is limited to ML/arXiv papers, so results and related searches will naturally skew toward ML topics even for unrelated queries.
- Difficulty classification reflects how the query *reads*, not the actual complexity of the returned paper.

## Possible Improvements

- Named Entity Recognition on search results
- Caching embeddings to avoid recomputing them on every run
- A simple web UI (Streamlit / Gradio) for interactive querying
- Expanding the corpus beyond ML papers for broader topic coverage

## Acknowledgements

Built using open-source tools from Hugging Face, Sentence-Transformers, Facebook AI (FAISS), and KeyBERT.
