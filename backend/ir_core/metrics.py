# backend/ir_core/metrics.py

from typing import List, Any, Set, Dict
import math

def precision(retrieved: List[Any], relevant: Set[Any]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / len(retrieved)

def recall(retrieved: List[Any], relevant: Set[Any]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved if d in relevant)
    return hits / len(relevant)

def f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def precision_at_k(retrieved: List[Any], relevant: Set[Any], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant)
    return hits / k

def dcg(relevances: List[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg(relevances: List[float], ideal_relevances: List[float]) -> float:
    ideal = dcg(sorted(ideal_relevances, reverse=True))
    if ideal == 0:
        return 0.0
    return dcg(relevances) / ideal

def evaluate_query(
    results: List[Dict[str, Any]],
    relevant_ids: Set[Any],
    k: int = 5,
) -> Dict[str, float]:
    retrieved_ids = [r["doc_id"] for r in results]
    p = precision(retrieved_ids, relevant_ids)
    r = recall(retrieved_ids, relevant_ids)
    f = f1_score(p, r)
    p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)

    rels = [1.0 if d in relevant_ids else 0.0 for d in retrieved_ids[:k]]
    ideal_rels = sorted(rels, reverse=True)
    dcg_val = dcg(rels)
    ndcg_val = ndcg(rels, ideal_rels)

    return {
        "precision": p,
        "recall": r,
        "f1": f,
        f"p@{k}": p_at_k,
        "dcg": dcg_val,
        "ndcg": ndcg_val,
    }
