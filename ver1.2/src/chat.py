import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트를 PYTHONPATH에 추가하여 패키지 임포트를 보장
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()
import ollama

from src.embeddings import EmbeddingStore
from src.retrieval import retrieve_documents
from src.message_manager import MessageManager

# Ollama 모델을 환경변수로 관리하되, 기본값 qwen3:latest 사용
model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# EmbeddingStore와 MessageManager 인스턴스 지연 초기화
store = None
msgManager = MessageManager()
msgManager.system_msg(
    "문서 내용을 기반으로 간결하게 **요약**해 답변하세요. 질문-답변 형식은 사용하지 마십시오."
)

def get_store():
    global store
    if store is None:
        store = EmbeddingStore()
    return store


def generate_answer(query, retrieved_docs, conversation_history):
    """
    이전 대화 기록에 사용자 메시지 추가
    generate_prompt() 로 시스템+문서+히스토리 프롬프트 생성
    ollama.chat() 스트리밍 출력
    완성된 답변 반환
    """
    msgManager.append_msg(query)
    msg = msgManager.generate_prompt(retrieved_docs)

    print("RAG : ", end="", flush=True)
    full_answer = ""
    for response in ollama.chat(model=model, messages=msg, stream=True):
        chunk = response.get("message", {}).get("content", "")
        print(chunk, end="", flush=True)
        full_answer += chunk
    print()

    return full_answer


def chat_loop(top_k: int = 2):
    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit' 입력):")
    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        docs = retrieve_documents(query, get_store(), top_k=top_k)
        if not docs:
            print("관련 문서를 찾을 수 없습니다.")
            continue

        answer = generate_answer(query, docs, msgManager.queue)
        msgManager.append_msg_by_assistant(answer)