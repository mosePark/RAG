## 프로젝트 개요
로컬 환경에서 PDF 문서를 벡터 검색(검색 증강 생성, RAG)을 통해 질의응답할 수 있는 챗봇입니다.  
- PDF → 텍스트(마크다운) 변환  
- 텍스트 분할 → GPT 임베딩 → elastic 색인  
- Ollama 로컬 LLM을 통해 검색된 청크 기반으로 답변 생성  

## 주요 기능
1. PDF 로더 (loader.py)  
2. 임베딩 생성 및 벡터 DB 저장 (embeddings.py)  
3. 검색 (retrieval.py)  
4. QA (chat.py / run.py)  


## 실행 방법

1. 인덱싱
```
python build_index.py
```

2. 챗봇 구동
```
ollama serve        # Ollama REST API 서버 실행
python run.py
```

## 디렉토리 구조
```
RAG/
├─ data/               # (gitignore) PDF 원본
├─ __init__.py
├─ build_index.py      # 색인 스크립트
├─ run.py              # 실행 진입점
├─ requirements.txt
├─ README.md
└─ src/
   ├─ __init__.py
   ├─ loader.py
   ├─ embeddings.py
   ├─ retrieval.py
   ├─ message_manager.py
   └─ chat.py
```

## 요구사항

```
ollama==0.4.8
python-dotenv==1.1.0
langchain==0.3.25
pymupdf4llm==0.0.24
elasticsearch==9.0.1
openai==1.78.1
tiktoken==0.9.0
```

## 문제점
- LLM inference 시간이 약 700s -> CPU 환경 문제
