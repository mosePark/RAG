import time
import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트를 PYTHONPATH에 추가하여 상대 임포트 보장
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 환경변수 로드
load_dotenv()

from src.chat import generate_answer, get_store, msgManager
from src.retrieval import retrieve_documents


def measure_time(query: str, top_k: int = 3) -> dict:
    """
    주어진 쿼리에 대해 Retrieval과 Generation 두 단계의 소요 시간을 측정합니다.
    반환값:
        {
            'retrieval_time': float (초),
            'generation_time': float (초),
            'doc_count': int,
            'answer_length': int
        }
    """
    # Retrieval 단계
    start_retrieval = time.time()
    store = get_store()
    docs = retrieve_documents(query, store, top_k=top_k)
    end_retrieval = time.time()

    # Generation 단계
    start_generation = time.time()
    answer = generate_answer(query, docs, msgManager.queue)
    end_generation = time.time()

    return {
        'retrieval_time': end_retrieval - start_retrieval,
        'generation_time': end_generation - start_generation,
        'doc_count': len(docs),
        'answer_length': len(answer)
    }

# 예시 사용법
# result = measure_time("측정할 질의 텍스트", top_k=2)
# print(result)
