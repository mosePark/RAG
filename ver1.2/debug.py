from elasticsearch import Elasticsearch
import os

# 1) 클라이언트 생성: timeout 제거
es = Elasticsearch(
    hosts=[{"host": "localhost", "port": 9200, "scheme": "https"}],
    basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
    verify_certs=False,    # self-signed cert 무시
    ssl_show_warn=False,   # 경고 숨기기
)

# 2) ping 시에 타임아웃 지정 (선택)
print("Ping OK?", es.ping(request_timeout=30))

# %%

# ——— 필수 임포트 ———
from src.embeddings import EmbeddingStore
from src.retrieval import retrieve_documents
from src.chat import generate_answer, msgManager

# ——— ES 스토어 생성 ———
# .env 로드(필요시)
from dotenv import load_dotenv
load_dotenv()

store = EmbeddingStore()  # index_name, 모델 등은 기본값 사용
query = "테스트 질의입니다."

# ——— 1) Elasticsearch 연결 및 검색 테스트 ———
try:
    docs = retrieve_documents(query, store, top_k=3)
    print("✅ Elasticsearch OK — 문서 수:", len(docs))
except Exception as e:
    print("❌ Elasticsearch 연결 오류:", type(e), e)
    raise

# ——— 2) Ollama 스트리밍 테스트 ———
try:
    # msgManager 는 src/chat.py 에서 생성된 전역 객체
    answer = generate_answer(query, docs, msgManager.queue)
    print("\n✅ Ollama OK — 응답 길이:", len(answer))
except Exception as e:
    print("❌ Ollama 연결 오류:", type(e), e)
    raise






# %%

# # test_components.py
# import os
# import ollama
# from dotenv import load_dotenv
# from src.embeddings import EmbeddingStore
# from src.retrieval import retrieve_documents

# # 0) .env 로드 (OLLAMA_SERVER, ES_HOST 등)
# load_dotenv()

# # 1) Elasticsearch 연결 및 검색 확인
# print("1) ES ping & search 테스트")
# es_store = None
# try:
#     es_store = EmbeddingStore()               # 기본 ES_HOST 읽어옴
#     print("   ES ping OK:", es_store.es.ping(request_timeout=10))
#     docs = retrieve_documents("테스트 질의", es_store, top_k=1)
#     print("   검색된 문서 수:", len(docs))
# except Exception as e:
#     print("   ❌ ES 관련 에러:", type(e).__name__, e)
#     raise SystemExit(1)

# # 2) 임베딩(EmbeddingStore.embedder.encode) 확인
# print("\n2) OpenAI 임베딩 테스트")
# try:
#     emb = es_store.embedder.encode(["hello world"], convert_to_tensor=False)
#     print("   임베딩 벡터 크기:", len(emb[0]))
# except Exception as e:
#     print("   ❌ 임베딩 에러:", type(e).__name__, e)
#     raise SystemExit(1)

# # 3) Ollama 채팅 스트리밍 확인
# print("\n3) Ollama 채팅 테스트")
# try:
#     # 최소한 시스템 메세지 한 개랑 user 메세지 한 개 넣어 보기
#     messages = [
#         {"role": "system", "content": "Say hello concisely."},
#         {"role": "user", "content": "안녕 Ollama?"}
#     ]
#     # 스트림 한 덩어리만 받자
#     for resp in ollama.chat(model=os.getenv("OLLAMA_MODEL","qwen3:latest"),
#                             server=os.getenv("OLLAMA_SERVER"),
#                             stream=True,
#                             messages=messages):
#         print("   >", resp["message"]["content"])
#         break
# except Exception as e:
#     print("   ❌ Ollama 에러:", type(e).__name__, e)
#     raise SystemExit(1)

# print("\n✅ 모든 컴포넌트 정상 동작합니다.")
