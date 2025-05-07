# src/retrieval.py

from .embeddings import EmbeddingStore

def retrieve_documents(query: str, store: EmbeddingStore, top_k=3) -> list[str]:
    q_emb = store.embedder.encode([query], convert_to_tensor=False)[0]
    res = store.collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    if not res["metadatas"]:
        return []
    return [m["text"] for m in res["metadatas"][0]]