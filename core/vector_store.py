# core/vector_store.py
"""
Simple vector store placeholder.
Provides basic in-memory storage of texts and simple retrieval by substring match.
Replace with FAISS/Pinecone/Chroma in production.
"""
from typing import List, Dict

class SimpleVectorStore:
    def __init__(self):
        self.docs = []

    def add(self, doc_id: str, text: str, metadata: dict = None):
        self.docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def search(self, query: str, top_k: int = 3):
        # naive substring match ranking
        scored = []
        for d in self.docs:
            score = d["text"].lower().count(query.lower().split()[0]) if d["text"] else 0
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k]]
