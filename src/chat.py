# src/chat.py

import ollama
from .embeddings import EmbeddingStore
from .retrieval import retrieve_documents
from .message_manager import MessageManager

model = "exaone3.5:2.4b"
store = EmbeddingStore()
msgManager = MessageManager()
msgManager.system_msg(
    "문서 내용을 기반으로 간결하게 **요약**해 답변하세요. 질문-답변 형식은 사용하지 마십시오."
)

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
        chunk = response["message"]["content"]
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

        docs = retrieve_documents(query, store, top_k=top_k)
        if not docs:
            print("관련 문서를 찾을 수 없습니다.")
            continue

        answer = generate_answer(query, docs, msgManager.queue)
        msgManager.append_msg_by_assistant(answer)