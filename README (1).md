# NLP Search System

A semantic search system built on ML research papers that goes beyond keyword matching — it searches by meaning, summarizes results, extracts keywords, judges query difficulty, and suggests related searches.

## Features

- **Semantic Search** — finds the most relevant papers using sentence embeddings, not just keyword overlap
- **Summarization** — condenses long abstracts into short, readable summaries
- **Keyword Extraction** — pulls out the most representative terms from a result
- **Difficulty Level** — classifies the query as Beginner, Intermediate, or Advanced
- **Related Searches** — suggests related topics pulled from the actual search results

## Tech Stack

| Component | Library / Model |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Search | `faiss` (cosine similarity via inner product) |
| Summarization | `transformers` (`facebook/bart-large-cnn`) |
| Keyword Extraction | `KeyBERT` |
| Difficulty Classification | `transformers` zero-shot classification (`facebook/bart-large-mnli`) |
| Dataset | `CShorten/ML-ArXiv-Papers` (via 🤗 `datasets`) |

## How It Works

1. **Search** — the query is embedded and compared against a FAISS index of paper embeddings to find the top matches.
2. **Summarize** — the top matching abstract is passed through a BART summarization model.
3. **Extract Keywords** — KeyBERT pulls the most relevant terms from that abstract.
4. **Classify Difficulty** — a zero-shot classifier scores the query against `beginner`, `intermediate`, and `advanced` labels.
5. **Suggest Related Searches** — KeyBERT runs again on the other top search results to surface related terms from the corpus.

## Example

```
full_search_query("medical research python fastapi")
```

```
Summary
...

Keywords
- imaging
- tomographic
- reconstruction

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
- api design
- authentication
- deep learning
```

## Setup

```bash
pip install datasets pandas sentence-transformers scikit-learn faiss-cpu
pip install transformers==4.46.3
pip install keybert==0.8.5
```

## Usage

Run the notebook cells in order, then call:

```python
full_search_query("your query here", k=5)
```

## Notes

- `facebook/bart-large-mnli` and `facebook/bart-large-cnn` are large models (~1.6GB each) — the first run will take time to download.
- Built and tested in Google Colab with a T4 GPU.

## Possible Improvements

- Named Entity Recognition on search results
- Caching embeddings to avoid recomputation
- A simple web UI for querying
