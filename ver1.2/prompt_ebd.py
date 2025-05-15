
from langchain.embeddings import OpenAIEmbeddings
import os

# 1) OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2) 임베딩 모델 초기화
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# 3) 질의문장 임베딩 생성
query = "법 내용 중 제 7조에 대해 설명해보시오."
q_vec = embedder.embed_query(query)

# 4) 벡터 출력 (리스트 형태)
print(q_vec)