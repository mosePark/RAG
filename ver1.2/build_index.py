from dotenv import load_dotenv
from os import getenv
from elasticsearch import Elasticsearch
from src.loader import load_pdf, split_text
from src.embeddings import EmbeddingStore

# 프로젝트 루트의 .env 파일 로드
load_dotenv()

def main():
    # 1) 대상 PDF 경로
    pdf_path = "data/국민건강보험법.pdf"

    # 2) PDF → 텍스트 → 청크 분할
    text = load_pdf(pdf_path)
    chunks = split_text(text)

    # 3) 검증된 Elasticsearch 클라이언트 생성
    es_client = Elasticsearch(
        hosts=[{"host": "localhost", "port": 9200, "scheme": "https"}],
        basic_auth=(getenv("ES_USER"), getenv("ES_PASSWORD")),
        verify_certs=False,    # self-signed cert 무시
        ssl_show_warn=False,   # SSL 경고 숨기기
    )
    print("ES Ping OK?", es_client.ping())

    # 4) ES 클라이언트를 주입하여 EmbeddingStore 생성
    store = EmbeddingStore(
        index_name="rag_index",
        embedding_model="text-embedding-3-small",
        dims=1536,
        es_client=es_client
    )

    # 5) 색인 실행
    store.index_chunks(chunks)
    print(f"✓ {len(chunks)}개 청크를 Elasticsearch에 색인 완료")

if __name__ == "__main__":
    main()