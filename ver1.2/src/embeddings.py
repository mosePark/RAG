# src/embeddings.py

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from elasticsearch import Elasticsearch, helpers

# 프로젝트 루트의 .env 파일 로드
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class EmbeddingStore:

    def __init__(
        self,
        index_name="rag_index",
        embedding_model="text-embedding-3-small",
        dims=1536,
        es_client=None,               # 외부에서 주입된 ES 클라이언트를 받을 수 있도록
    ):
        # 1) ES 클라이언트 설정
        if es_client:
            self.es = es_client
        else:
            es_host     = os.getenv("ES_HOST", "https://localhost:9200")
            # print(f"[DEBUG] → ES_HOST from env: {es_host!r}") # 디버깅 코드 한줄 나중에 지우기기
            es_user     = os.getenv("ES_USER")
            es_password = os.getenv("ES_PASSWORD")
            auth = (es_user, es_password) if es_user and es_password else None
            self.es = Elasticsearch(
                [es_host],
                basic_auth=auth,
                verify_certs=False,
                ssl_show_warn=False,
            )

        self.index = index_name
        self.dims  = dims

        # 2) 인덱스 없으면 생성
        if not self.es.indices.exists(index=self.index):
            mapping = {
                "mappings": {
                    "properties": {
                        "text":      {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": self.dims},
                        "metadata":  {"type": "object"},
                    }
                }
            }
            self.es.indices.create(index=self.index, body=mapping)

        # 3) LangChain OpenAIEmbeddings 초기화
        #    파라미터 이름이 model_name이 아니라 model 입니다!
        self.embedder = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def index_chunks(self, chunks, metadatas=None):
        # 1) 청크별 임베딩 생성
        embeddings = self.embedder.embed_documents(chunks)

        # 2) bulk 색인용 액션 준비
        actions = []
        for i, emb in enumerate(embeddings):
            meta = metadatas[i] if metadatas else {}
            actions.append({
                "_index": self.index,
                "_id":      f"chunk_{i}",
                "_source": {
                    "text":      chunks[i],
                    "embedding": emb,
                    "metadata":  meta,
                }
            })

        # 3) bulk 색인 수행 및 인덱스 refresh
        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index)
