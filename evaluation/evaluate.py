from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _embed(text: str):
    return embedder.encode([text])[0]


def _cos_sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])


def evaluate_answer(
    query: str,
    answer: str,
    context: str,
    docs: List
) -> Dict:
    """
    Improved evaluation:
    - Answer ↔ Doc similarity (grounding)
    - Query ↔ Doc similarity (relevance)
    - Combined score
    """

    if not docs:
        return {
            "confidence": 0.0,
            "verdict": "No Context"
        }

    query_emb = _embed(query)
    answer_emb = _embed(answer)

    doc_scores = []

    for doc in docs:
        doc_text = doc.page_content[:500]  # trim long chunks

        doc_emb = _embed(doc_text)

        # 🔹 Grounding: answer vs doc
        grounding = _cos_sim(answer_emb, doc_emb)

        # 🔹 Relevance: query vs doc
        relevance = _cos_sim(query_emb, doc_emb)

        # 🔹 Combined score (weighted)
        score = 0.7 * grounding + 0.3 * relevance

        doc_scores.append(score)

    # 🔹 Aggregate smarter than max
    top_k_scores = sorted(doc_scores, reverse=True)[:3]

    final_score = float(np.mean(top_k_scores)) if top_k_scores else 0.0

    # 🔹 Verdict mapping (tighter)
    if final_score > 0.75:
        verdict = "Highly Grounded"
    elif final_score > 0.6:
        verdict = "Likely Grounded"
    elif final_score > 0.4:
        verdict = "Weakly Grounded"
    else:
        verdict = "Likely Hallucinated"

    return {
        "confidence": round(final_score, 3),
        "verdict": verdict
    }





