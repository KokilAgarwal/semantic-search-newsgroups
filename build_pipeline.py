from data.load_dataset import load_data
from embeddings.generate_embeddings import generate_embeddings
from vector_db.faiss_index import build_index
import faiss

print("Loading dataset...")
texts, labels = load_data()

print("Generating embeddings...")
embeddings = generate_embeddings(texts)

print("Building FAISS index...")
index = build_index(embeddings)

faiss.write_index(index, "vector_db/news_index.faiss")

print("Vector database saved.")