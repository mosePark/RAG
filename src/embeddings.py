# src/embeddings.py

from sentence_transformers import SentenceTransformer
import chromadb

class EmbeddingStore:
    def __init__(self, db_path: str = "./db/chroma_db"):
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )


    def index_chunks(self, chunks: list[str]):
        """
        청크 목록을 임베딩 후 DB에 저장
        """
        embeddings = self.embedder.encode(chunks, convert_to_tensor=False)
        for i, emb in enumerate(embeddings):
            self.collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[emb.tolist()],
                metadatas=[{"text": chunks[i]}]
            )