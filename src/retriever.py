import os
import re
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# ---------- CONFIG ----------
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 4

# ---------- LOAD EMBEDDINGS ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- LOAD VECTOR STORE ----------
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

vectorstore = FAISS.load_local(
    INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# ---------- RERANKER ----------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------- QUERY NORMALIZATION ----------
def normalize_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r'[^a-z0-9\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query


# ---------- RERANK ----------
def rerank(query, docs, top_k=TOP_K_RERANK):
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print("\n[DEBUG] Rerank Scores:")
    for doc, score in ranked[:top_k]:
        print(f"{score:.4f} | {doc.metadata.get('source')}")

    return [doc for doc, _ in ranked[:top_k]]


# ---------- MAIN RETRIEVAL ----------
def retrieve(query, use_rerank=True):
    # Step 1: normalize query
    query_clean = normalize_query(query)

    # Step 2: retrieve candidates
    results = vectorstore.similarity_search_with_score(
        query_clean, k=TOP_K_RETRIEVE
    )

    if not results:
        return []

    # Step 3: debug raw scores
    print("\n[DEBUG] Raw Retrieval Scores:")
    for doc, score in results:
        print(f"{score:.4f} | {doc.metadata.get('source')}")

    docs = [doc for doc, _ in results]

    # Step 4: rerank FIRST (important)
    if use_rerank:
        docs = rerank(query_clean, docs)

    # Step 5: dynamic filtering (after rerank)
    best_score = results[0][1]
    dynamic_threshold = best_score * 1.5

    filtered_docs = [
        doc for doc, score in results if score <= dynamic_threshold
    ]

    # fallback safety
    if not filtered_docs:
        print("\n[INFO] Using best available match (fallback).")
        filtered_docs = [results[0][0]]

    # final selection
    final_docs = filtered_docs[:TOP_K_RERANK]

    return final_docs


# ---------- DEBUG TEST ----------
if __name__ == "__main__":
    test_queries = [
        "What are hallucinations in LLMs?",
        "Explain gradient descent update rule",
        "what about its types?"
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")

        results = retrieve(query)

        if not results:
            print("\nNo relevant information found.")
        else:
            for i, d in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(d.page_content[:300])
                print("Source:", d.metadata.get("source"))
                print("Page:", d.metadata.get("page"))




# import os
# import warnings
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import CrossEncoder

# warnings.filterwarnings("ignore")

# # ---------- PATH SETUP ----------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# # ---------- CONFIG ----------
# TOP_K_RETRIEVE = 8
# TOP_K_RERANK = 4

# SIMILARITY_THRESHOLD = 0.8     # strict filter (good matches)
# FALLBACK_THRESHOLD = 1       # allow 1 weak match if somewhat relevant

# # ---------- LOAD EMBEDDINGS ----------
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # ---------- LOAD VECTOR STORE ----------
# if not os.path.exists(INDEX_PATH):
#     raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

# vectorstore = FAISS.load_local(
#     INDEX_PATH,
#     embedding_model,
#     allow_dangerous_deserialization=True
# )

# # ---------- RERANKER ----------
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# # ---------- RERANK FUNCTION ----------
# def rerank(query, docs, top_k=TOP_K_RERANK):
#     if not docs:
#         return []

#     pairs = [(query, d.page_content) for d in docs]
#     scores = reranker.predict(pairs)

#     ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

#     return [doc for doc, _ in ranked[:top_k]]


# # ---------- MAIN RETRIEVAL ----------
# def retrieve(query, use_rerank=True):
#     # Step 0: basic query cleanup
#     query = query.lower().strip()

#     # Step 1: retrieve with scores
#     results = vectorstore.similarity_search_with_score(
#         query, k=TOP_K_RETRIEVE
#     )

#     if not results:
#         return []

#     # Step 2: debug raw scores
#     print("\n[DEBUG] Raw Retrieval Scores:")
#     for doc, score in results:
#         print(f"{score:.4f} | {doc.metadata.get('source')}")

#     # Step 3: strict filtering
#     filtered = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

#     # Step 4: soft fallback (if nothing passes)
#     if not filtered:
#         best_doc, best_score = results[0]

#         if best_score < FALLBACK_THRESHOLD:
#             print("\n[INFO] Using best available match (soft fallback).")
#             filtered = [(best_doc, best_score)]
#         else:
#             print("\n[INFO] No relevant documents found.")
#             return []

#     docs = [doc for doc, _ in filtered]

#     # Step 5: rerank
#     if use_rerank:
#         docs = rerank(query, docs)

#     return docs


# # ---------- DEBUG TEST ----------
# if __name__ == "__main__":
#     test_queries = [
#         "What are hallucinations in LLMs?",
#         "Explain gradient descent update rule",
        
#     ]

#     for query in test_queries:
#         print("\n" + "=" * 60)
#         print(f"Query: {query}")

#         results = retrieve(query)

#         if not results:
#             print("\nNo relevant information found in the knowledge base.")
#         else:
#             for i, d in enumerate(results):
#                 print(f"\n--- Result {i+1} ---")
#                 print(d.page_content[:300])
#                 print("Source:", d.metadata.get("source"))
#                 print("Page:", d.metadata.get("page"))