# backend/ir_core/indexers.py

from typing import List, Any, Literal
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .tokenization import TOKENIZER_REGISTRY
from .preprocessing import PreprocessingConfig
from . import rankers

IndexType = Literal["tf", "tfidf", "bm25"]

@dataclass
class IndexConfig:
    name: str
    tokenizer_name: str
    remove_stopwords: bool
    use_stemming: bool
    index_type: IndexType

class Index:
    def __init__(self, config: IndexConfig, docs: List[str], doc_ids: List[Any]):
        self.config = config
        self.docs = docs
        self.doc_ids = doc_ids
        self.vectorizer = None
        self.doc_matrix = None
        self.idf_ = None  # for BM25

    def _build_analyzer(self):
        tokenizer = TOKENIZER_REGISTRY[self.config.tokenizer_name]
        prep = PreprocessingConfig(
            remove_stopwords=self.config.remove_stopwords,
            use_stemming=self.config.use_stemming,
        )

        def analyzer(text: str):
            tokens = tokenizer.tokenize(text)
            tokens = prep.apply(tokens)
            return tokens

        return analyzer

    def build(self):
        analyzer = self._build_analyzer()

        if self.config.index_type in {"tf", "bm25"}:
            self.vectorizer = CountVectorizer(analyzer=analyzer)
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(analyzer=analyzer)

        self.doc_matrix = self.vectorizer.fit_transform(self.docs)

        if self.config.index_type == "bm25":
            # compute IDF for BM25
            # (df = number of docs containing each term)
            binarized = (self.doc_matrix > 0).astype(int)
            df = np.array(binarized.sum(axis=0)).ravel()
            n_docs = self.doc_matrix.shape[0]
            self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5) + 1e-8)

    def search(self, query: str, top_k: int = 10):
        analyzer = self._build_analyzer()
        q_tokens = analyzer(query)
        q_text = " ".join(q_tokens)
        q_vec = self.vectorizer.transform([q_text])

        if self.config.index_type == "tf":
            scores = rankers.rank_tf(q_vec, self.doc_matrix)
        elif self.config.index_type == "tfidf":
            scores = rankers.rank_tfidf(q_vec, self.doc_matrix)
        else:
            scores = rankers.rank_bm25(q_vec, self.doc_matrix, self.idf_)

        sorted_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in sorted_idx:
            doc_text = self.docs[i]
            snippet = doc_text[:400].replace("\n", " ").strip()
            results.append(
                {
                    "doc_id": self.doc_ids[i],
                    "score": float(scores[i]),
                    "snippet": snippet + ("..." if len(doc_text) > 400 else ""),
                }
            )
        return results
