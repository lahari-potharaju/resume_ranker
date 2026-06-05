# backend/app.py

from pathlib import Path
import sys
from typing import List, Literal, Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure local ir_core package is importable when running from the repo root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ir_core.corpus import Corpus
from ir_core.indexers import IndexConfig, Index
from ir_core.experiments import ExperimentRunner, Query

app = FastAPI(title="IRLab - Resume IR System")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

# --- Load corpus at startup ---

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "resumes.csv"

# Adjust text_col to your CSV column name
corpus = Corpus(
    name="kaggle_resumes",
    path=DATA_PATH,
    text_col="Resume_str",  # change this if your CSV column differs
    id_col=None,
)

docs = corpus.documents()
DOC_IDS = [d.id for d in docs]
DOC_TEXTS = [d.text for d in docs]


# --- Pydantic request models ---

class SearchRequest(BaseModel):
    query: str
    tokenizer_name: Literal["simple", "resume_aware"] = "resume_aware"
    remove_stopwords: bool = True
    use_stemming: bool = False
    index_type: Literal["tf", "tfidf", "bm25"] = "tfidf"
    top_k: int = 10


class ExperimentRequest(BaseModel):
    queries: List[str]
    relevant_ids_list: List[List[Any]]
    tokenizer_name: Literal["simple", "resume_aware"] = "resume_aware"
    remove_stopwords: bool = True
    use_stemming: bool = False
    index_type: Literal["tf", "tfidf", "bm25"] = "tfidf"
    top_k: int = 10


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"message": "IRLab backend running"}


@app.post("/search")
def search(req: SearchRequest):
    config = IndexConfig(
        name="ad_hoc",
        tokenizer_name=req.tokenizer_name,
        remove_stopwords=req.remove_stopwords,
        use_stemming=req.use_stemming,
        index_type=req.index_type,
    )
    index = Index(config, DOC_TEXTS, DOC_IDS)
    index.build()
    results = index.search(req.query, top_k=req.top_k)
    return {"config": config.__dict__, "results": results}


@app.post("/experiment")
def experiment(req: ExperimentRequest):
    config = IndexConfig(
        name="experiment_config",
        tokenizer_name=req.tokenizer_name,
        remove_stopwords=req.remove_stopwords,
        use_stemming=req.use_stemming,
        index_type=req.index_type,
    )

    if len(req.queries) != len(req.relevant_ids_list):
        raise ValueError("queries and relevant_ids_list must have same length")

    q_objs = [
        Query(text=q, relevant_ids=rids)
        for q, rids in zip(req.queries, req.relevant_ids_list)
    ]

    runner = ExperimentRunner(DOC_TEXTS, DOC_IDS)
    report = runner.run(config, q_objs, top_k=req.top_k)
    return report
