## 프로젝트 개요
로컬 환경에서 PDF 문서를 벡터 검색(검색 증강 생성, RAG)을 통해 질의응답할 수 있는 챗봇입니다.  
- PDF → 텍스트(마크다운) 변환  
- 텍스트 분할 → 임베딩 → ChromaDB 색인  
- Ollama 로컬 LLM을 통해 검색된 청크 기반으로 답변 생성  

## 주요 기능
1. PDF 로딩 및 청킹 (loader.py)  
2. 임베딩 생성 및 벡터 DB 저장 (embeddings.py)  
3. 검색 (retrieval.py)  
4. 대화 루프 (chat.py / run.py)  

## 설치 방법

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
├─ db/                 # (gitignore) ChromaDB 데이터
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
chromadb==1.0.8
langchain==0.3.24
ollama==0.4.8
pymupdf4llm==0.0.22
sentence-transformers==4.1.0
```

