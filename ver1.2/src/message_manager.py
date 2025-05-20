# src/message_manager.py

from collections import deque
import tiktoken

def drop_token(doc: str, max_tokens: int = 300, model: str = "gpt-3.5-turbo") -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(doc)
    return enc.decode(tokens[:max_tokens])

class MessageManager:
    def __init__(self, system_prompt: str = "", max_history: int = 10):
        self._system_msg = {"role": "system", "content": system_prompt}
        self.queue = deque(maxlen=max_history)

    def create_msg(self, role, content):
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        msg = self.create_msg("user", content)
        self.queue.append(msg)

    def append_msg_by_assistant(self, content):
        msg = self.create_msg("assistant", content)
        self.queue.append(msg)

    def generate_prompt(self, retrieved_docs: list[str], doc_token_limit: int = 300) -> list[dict]:
        # 문서 토큰 자르기
        model = "gpt-3.5-turbo"
        limited_docs = [
            drop_token(doc, max_tokens=doc_token_limit, model=model)
            for doc in retrieved_docs
        ]
        docs_combined = "\n\n".join(limited_docs)

        # 가장 마지막 user 질문만 추출
        user_question = next((msg["content"] for msg in reversed(self.queue) if msg["role"] == "user"), "질문 없음")

        return [
            self._system_msg,
            {
                "role": "user",
                "content": f"[문서 내용]\n{docs_combined}"
            },
            {
                "role": "user",
                "content": f"[질문]\n{user_question}"
            }
        ]


msgManager = MessageManager()
msgManager.system_msg(
    "당신은 문서 기반 응답 시스템입니다. '문서 내용'을 참고해 '질문'에 대해 한국어로 정확하고 간결하게 요약된 답변을 하세요. 질문이 명확하지 않더라도 문서를 기준으로 최대한 구체적으로 설명하세요."
)