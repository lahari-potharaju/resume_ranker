# AI Resume Ranking System

An information retrieval system that ranks resumes against a job description using classical IR techniques — TF, TF-IDF, and BM25. Built with a FastAPI backend and a vanilla JS frontend.

## Features

- Paste any job description and instantly rank matching resumes
- Three ranking strategies: **TF**, **TF-IDF**, **BM25**
- Two tokenizers: **Simple** (regex) and **Resume-aware** (merges common job phrases like `machine_learning`, `data_scientist`)
- Optional stopword removal and Porter stemming
- Experiment mode: supply known-relevant resume IDs and get **Precision, Recall, F1, P@K, DCG, NDCG** metrics

## Project Structure

```
resume_ranker/
├── backend/
│   ├── app.py              # FastAPI app — routes and startup
│   ├── requirements.txt    # Python dependencies
│   └── ir_core/            # Core IR library
│       ├── corpus.py       # CSV loader → Document objects
│       ├── indexers.py     # Builds TF / TF-IDF / BM25 index
│       ├── rankers.py      # Scoring functions (TF dot, cosine, BM25)
│       ├── tokenization.py # Simple and resume-aware tokenizers
│       ├── preprocessing.py# Stopword removal + Porter stemming
│       ├── experiments.py  # Runs multi-query evaluation
│       └── metrics.py      # Precision, Recall, F1, DCG, NDCG
├── frontend/
│   ├── index.html          # Single-page UI
│   ├── app.js              # Fetch calls to /search and /experiment
│   └── style.css           # Dark-theme styling
├── data/
│   └── resumes.csv         # Kaggle resume dataset (Resume_str column)
├── docs/
│   └── IRLab_Story_Mode_Document.docx  # Project report / story doc
└── .gitignore
```

## Getting Started

### Prerequisites
- Python 3.11+

### Setup

```bash
cd backend
python -m venv winenv
winenv\Scripts\activate        # Windows
# source winenv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Run

```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the frontend UI |
| GET | `/health` | Health check |
| POST | `/search` | Rank resumes for a job description |
| POST | `/experiment` | Evaluate ranking with relevance labels |

### POST `/search` — example payload

```json
{
  "query": "Python software engineer with machine learning experience",
  "tokenizer_name": "resume_aware",
  "remove_stopwords": true,
  "use_stemming": false,
  "index_type": "tfidf",
  "top_k": 5
}
```

### POST `/experiment` — example payload

```json
{
  "queries": ["data engineer", "machine learning engineer"],
  "relevant_ids_list": [[10, 42, 87], [5, 19]],
  "tokenizer_name": "resume_aware",
  "remove_stopwords": true,
  "use_stemming": false,
  "index_type": "bm25",
  "top_k": 10
}
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **IR:** scikit-learn (vectorization), NumPy, SciPy
- **NLP:** NLTK (Porter stemmer), custom tokenizers
- **Frontend:** HTML, CSS, Vanilla JS (no frameworks)
- **Data:** Kaggle Resume Dataset
