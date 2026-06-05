# backend/ir_core/rankers.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def rank_tf(q_vec: csr_matrix, doc_matrix: csr_matrix) -> np.ndarray:
    """
    TF ranking: dot product between query TF and doc TF matrix.
    """
    return (doc_matrix @ q_vec.T).toarray().ravel()

def rank_tfidf(q_vec: csr_matrix, doc_matrix: csr_matrix) -> np.ndarray:
    """
    TF-IDF ranking: cosine similarity.
    """
    return cosine_similarity(q_vec, doc_matrix).ravel()

def rank_bm25(
    q_vec: csr_matrix,
    doc_matrix: csr_matrix,
    idf: np.ndarray,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """
    Simplified BM25 implementation over a sparse CountVectorizer matrix.
    Only query terms contribute to the score.
    """
    csr = doc_matrix.tocsr()
    n_docs = csr.shape[0]

    doc_len = np.array(csr.sum(axis=1)).ravel()
    avg_dl = doc_len.mean() + 1e-8

    scores = np.zeros(n_docs, dtype=float)

    q_indices = q_vec.nonzero()[1]
    if q_indices.size == 0:
        return scores

    q_data = q_vec.toarray().ravel()

    for term_index in q_indices:
        tf = csr[:, term_index].toarray().ravel()
        if np.all(tf == 0):
            continue

        idf_term = idf[term_index]
        numer = tf * (k1 + 1.0)
        denom = tf + k1 * (1.0 - b + b * doc_len / avg_dl)
        scores += idf_term * (numer / (denom + 1e-8)) * q_data[term_index]

    return scores
