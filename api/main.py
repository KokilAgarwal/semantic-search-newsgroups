from fastapi import FastAPI
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from cache.semantic_cache import SemanticCache
from data.load_dataset import load_data
from embeddings.generate_embeddings import generate_embeddings

app = FastAPI()

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("vector_db/news_index.faiss")

print("Loading dataset...")
texts, labels = load_data()

cache = SemanticCache()


def search_vector_db(query_embedding, k=5):

    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(texts[idx][:300])

    return results


@app.post("/query")
def query_api(data: dict):

    query_text = data["query"]

    query_embedding = model.encode(query_text)

    cache_result = cache.lookup(query_embedding)

    if cache_result["hit"]:

        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["score"],
            "result": cache_result["result"],
            "dominant_cluster": cache_result["cluster"]
        }

    results = search_vector_db(query_embedding)

    cluster_id = 0

    cache.add(query_text, query_embedding, results, cluster_id)

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}