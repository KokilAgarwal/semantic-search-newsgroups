from data.load_dataset import load_data
from embeddings.generate_embeddings import generate_embeddings
from vector_db.faiss_index import build_index
from clustering.fuzzy_clustering import perform_clustering

import faiss
import numpy as np

print("Loading dataset...")
texts, labels = load_data()

print("Generating embeddings...")
embeddings = generate_embeddings(texts)

print("Building FAISS index...")
index = build_index(embeddings)

faiss.write_index(index, "vector_db/news_index.faiss")

print("Running clustering...")
gmm, probs = perform_clustering(embeddings)

np.save("clustering/cluster_probabilities.npy", probs)

print("Clustering complete.")