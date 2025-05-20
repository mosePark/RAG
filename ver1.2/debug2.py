# %% 챗 디버깅

# %% 라이브러리 임포트
import ollama
from ollama import ChatResponse, Message

# %% 질문 및 모델 설정
question = input("질문을 입력하세요: ")
model_name = "qwen3:4b"
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content=question)
]

# I%% 모델에 단일 응답 요청
response: ChatResponse = ollama.chat(model=model_name, messages=messages, stream=False)

# %% 응답 내용 확인 및 출력
if hasattr(response, 'message') and response.message:
    answer = response.message.content
    print("Answer:", answer)
else:
    # response.error 필드 또는 기타 속성 확인
    error_msg = getattr(response, 'error', None)
    print("[ERROR]", error_msg or "응답을 받지 못했습니다.")

# %% 디버깅용 전체 결과 출력
print("[DEBUG] ollama.chat 반환값:", response)

# %% 엘라스틱 서치 디버깅

# %%

from src.embeddings import EmbeddingStore
from src.retrieval import retrieve_documents

query = "공단의 관장 업무를 나열하라"
store = EmbeddingStore()
q_emb = store.embedder.embed_query(query)

docs = retrieve_documents(query, store, top_k=3)
docs

# %% 시간 재보기

import time
from src.embeddings import EmbeddingStore
from src.retrieval import retrieve_documents
from src.chat import generate_answer, msgManager

query = "국민건강보험공단의 업무에 대해 정리해줘."
store = EmbeddingStore()

# 1. 문서 검색 시간 측정
start = time.time()
docs = retrieve_documents(query, store, top_k=3)
print(f"\n⏱️ 문서 검색 시간: {time.time() - start:.3f}초")

# 2. 프롬프트 생성 시간 측정
start = time.time()
msgManager.append_msg(query)  # 사용자 메시지 추가
msg = msgManager.generate_prompt(docs)
print(f"⏱️ 프롬프트 생성 시간: {time.time() - start:.3f}초")

# 3. 모델 응답 시간 측정
start = time.time()
full_answer = ""
print("RAG 응답 (stream): ", end="", flush=True)
import ollama
model = "qwen3:4b"  # 환경 변수에서 가져오거나 직접 설정

for response in ollama.chat(model=model, messages=msg, stream=True):
    if hasattr(response, "message") and response.message:
        content = response.message.content or ""
    else:
        content = response.get("message", {}).get("content", "")
    print(content, end="", flush=True)
    full_answer += content

print(f"\n⏱️ 모델 응답 시간: {time.time() - start:.3f}초")

# 4. 결과 요약
print("\n✅ 전체 요약 완료")

# %% 병목 지점 식별

import tiktoken

def count_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """
    messages: List of dicts like [{"role": "system", "content": "..."}, ...]
    """
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += 4  # role & formatting tokens (OpenAI 기준)
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 2  # reply primer
    return total_tokens

#%%

# 프롬프트 생성
msgManager.append_msg(query)  # 사용자 질문 추가
prompt = msgManager.generate_prompt(docs)

# 토큰 수 측정
tokens = count_tokens_from_messages(prompt)
print(f"📏 총 프롬프트 토큰 수: {tokens} tokens")
# %%

print("📄 문서 길이:")
for i, doc in enumerate(docs):
    print(f"[{i+1}] {len(doc)}자 / {len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(doc))} tokens")
