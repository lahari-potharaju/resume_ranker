# backend/ir_core/experiments.py

from typing import List, Any, Dict
from dataclasses import dataclass

from .indexers import IndexConfig, Index
from .metrics import evaluate_query

@dataclass
class Query:
    text: str
    relevant_ids: List[Any]

class ExperimentRunner:
    def __init__(self, docs: List[str], doc_ids: List[Any]):
        self.docs = docs
        self.doc_ids = doc_ids

    def run(
        self,
        index_config: IndexConfig,
        queries: List[Query],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        index = Index(index_config, self.docs, self.doc_ids)
        index.build()

        per_query_metrics = []
        for q in queries:
            results = index.search(q.text, top_k=top_k)
            m = evaluate_query(results, set(q.relevant_ids), k=top_k)
            per_query_metrics.append(m)

        # aggregate
        keys = per_query_metrics[0].keys() if per_query_metrics else []
        aggregated = {
            k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics)
            for k in keys
        }

        return {
            "config": index_config.__dict__,
            "per_query": per_query_metrics,
            "aggregated": aggregated,
        }
