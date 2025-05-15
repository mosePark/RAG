# src/message_manager.py

from collections import deque

class MessageManager:
    def __init__(self, system_prompt: str = "", max_history: int = 10):
        # system_prompt 가 넘어오면 바로 설정
        self._system_msg = {"role": "system", "content": system_prompt}
        self.queue = deque(maxlen=max_history) # 대화 최대 몇개 저장하는지

    def create_msg(self, role, content):
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        msg = self.create_msg("user", content)
        self.queue.append(msg)

    def get_chat(self):
        return [self._system_msg] + list(self.queue)

    def set_retrived_docs(self, docs):
        self.retrieved_docs = docs

    def append_msg_by_assistant(self, content):
        msg = self.create_msg("assistant", content)
        self.queue.append(msg)

    def generate_prompt(self, retrieved_docs):

        docs = "\n".join(retrieved_docs)

        prompt = [msgManager._system_msg,{
            "role": "system",
            "content": f"문서 내용: {docs}\n질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오.",
        }] + list(msgManager.queue)

        return prompt


msgManager = MessageManager() # 객체 생성

msgManager.system_msg(
    "가장 마지막 'user'의 'content'에 대해 답변한다."
    "질문에 답할 때는 'system' 메시지 중 '문서 내용'에 명시된 부분을 우선 참고하여 정확히 답한다."
    "개행은 문장이 끝날때와 서로 다른 주제나 항목을 구분할 때 사용하며, 불필요한 개행은 넣지 않는다."
)