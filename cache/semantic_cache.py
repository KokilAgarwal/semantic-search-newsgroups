import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0


    def lookup(self, query_embedding):

        best_match = None
        best_score = 0

        for entry in self.cache:

            score = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.threshold:

            self.hit_count += 1

            return {
                "hit": True,
                "matched_query": best_match["query"],
                "result": best_match["result"],
                "score": float(best_score),
                "cluster": best_match["cluster"]
            }

        self.miss_count += 1
        return {"hit": False}


    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })


    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = 0
        if total > 0:
            hit_rate = self.hit_count / total

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }


    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0