Semantic Search System with Fuzzy Clustering and Semantic Cache
This project implements a lightweight semantic search system built using the 20 Newsgroups dataset. The goal is to demonstrate how machine learning techniques and efficient system design can be combined to create an intelligent search system.
Instead of relying on keyword matching, this system uses semantic embeddings to understand the meaning of queries and retrieve relevant documents.

Dataset
The dataset used is the Twenty Newsgroups dataset, available from the UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
It contains roughly 20,000 documents across 20 different discussion topics, making it a useful benchmark for text processing and information retrieval tasks.

Key Components
The system is composed of several core components:
Embedding Generation
Documents are converted into dense vector representations using the Sentence Transformer model:
  all-MiniLM-L6-v2
These embeddings capture the semantic meaning of each document.

Vector Database
The embeddings are indexed using FAISS, which allows fast similarity search across high-dimensional vectors.

Fuzzy Clustering
Gaussian Mixture Models are used to perform fuzzy clustering, allowing documents to belong to multiple clusters with different probabilities.

Semantic Cache
The system includes a custom semantic cache that identifies similar queries based on embedding similarity. If a new query closely resembles a previously processed query, the cached result is reused.

FastAPI Service
The system is exposed through a FastAPI service, allowing users to interact with the search engine through REST endpoints.

API Endpoints
  POST /query - Accepts a natural language query and returns semantically relevant documents.
  GET /cache/stats - Returns statistics about cache usage.
  DELETE /cache - Clears the cache.

Running the Project
Install dependencies: pip install -r requirements.txt
Build embeddings and vector database: python build_pipeline.py
Start the API server: uvicorn api.main:app --reload
Open the API documentation:
  http://127.0.0.1:8000/docs
  
Author
Kokil Agarwal
