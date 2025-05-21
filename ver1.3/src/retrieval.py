# src/retrieval.py

from .embeddings import EmbeddingStore

def retrieve_documents(
    query: str,
    store: EmbeddingStore,
    top_k: int = 3
) -> list[str]:
    """
    1) 사용자 쿼리를 임베딩으로 변환
    2) Elasticsearch에서 cosineSimilarity + 1.0으로 유사도 검색
    3) 텍스트만 추출하여 반환
    """
    # 1) 쿼리 임베딩
    # OpenAIEmbeddings에서는 embed_query 또는 embed_documents를 사용합니다.
    q_emb = store.embedder.embed_query(query)

    # 2) Elasticsearch 유사도 검색 (cosineSimilarity + 1.0)
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": q_emb}
                }
            }
        }
    }
    res = store.es.search(index=store.index, body=body)

    # 3) 결과에서 텍스트만 추출하여 반환
    hits = res.get("hits", {}).get("hits", [])
    return [hit["_source"]["text"] for hit in hits]