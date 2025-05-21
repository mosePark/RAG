import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """
    PDF → Markdown → 통합 텍스트
    """
    pdf_data = pymupdf4llm.to_markdown(file_path)
    text = "".join(pdf_data)
    print("PDF파일 로드")
    return text

def split_text(text):
    """
    텍스트를 청킹 후 리스트 반환
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# %%

from dotenv import load_dotenv
from os import getenv
from elasticsearch import Elasticsearch
from src.loader import load_pdf, split_text
from src.embeddings import EmbeddingStore

# 프로젝트 루트의 .env 파일 로드
load_dotenv()

# 1) 대상 PDF 경로
pdf_path = "data/국민건강보험법.pdf"

# 2) PDF → 텍스트 → 청크 분할
text = load_pdf(pdf_path)
chunks = split_text(text)

chunks.head()

# %% 새 전략략

import re
from typing import List, Dict
import pdfplumber

# 1. PDF 로드
def load_pdf(file_path: str) -> str:
    """
    PDF 파일에서 텍스트를 줄 단위로 추출 (조문 구분 보존에 유리)
    """
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    print("PDF 로드 완료")
    return full_text

# 2. 법령 텍스트를 JSON 구조로 파싱
def split_text(text: str) -> List[Dict]:
    """
    법령 텍스트를 조-항-호-목 구조로 파싱
    """
    articles = _split_by_article(text)
    return [_parse_article(a) for a in articles if a.strip()]

# 내부 함수: 조문 단위로 분할
def _split_by_article(text: str) -> List[str]:
    # 멀티라인 모드에서 줄 시작(^) 기준으로, 제목(괄호) 포함된 조문 앞에서만 분리
    return re.split(r'(?m)(?=^제\d+조(?:의\d+)?\([^)]*\))', text)

# 내부 함수: 각 조문을 계층 구조로 파싱
def _parse_article(raw: str) -> Dict:
    # 줄 시작의 조문 번호와 괄호 안 제목만 추출
    match = re.match(r'^(제\d+조(?:의\d+)?)(\([^)]*\))', raw, flags=re.MULTILINE)
    if match:
        number = match.group(1).strip()
        title = match.group(2).strip("()")
        body = raw[match.end():]
    else:
        number = "UNKNOWN"
        title = ""
        body = raw

    clauses = _split_clauses(body)
    return {
        "article_number": number,
        "title": title,
        "clauses": clauses
    }

# 내부 함수: 항(①-⑳) 분할 및 하위 구조 포함
def _split_clauses(text: str) -> List[Dict]:
    text = text.strip()
    # 항 표기가 없으면 전체 본문을 단일 항으로 처리
    if not re.search(r'[①-⑳]', text):
        return [{
            "clause_text": text,
            "items": _split_items(text)
        }]

    raws = re.split(r'(?=[①-⑳])', text)
    clauses = []
    for clause in raws:
        clause = clause.strip()
        if not clause:
            continue
        items = _split_items(clause)
        clauses.append({
            "clause_text": clause,
            "items": items
        })
    return clauses

# 내부 함수: 호 및 목 분할
def _split_items(text: str) -> List[Dict]:
    # 줄바꿈+숫자+점 패턴으로만 호 분리
    items_raw = re.split(r'(?=\n\d+\.)', text)
    items = []
    for item in items_raw:
        item = item.strip()
        if not item:
            continue
        # 줄바꿈+가-하+점 패턴으로만 목 분리
        sub = re.split(r'(?=\n[가-하]\.)', item)
        sub_items = [s.strip() for s in sub if s.strip()]
        items.append({
            "item_text": item,
            "sub_items": sub_items if len(sub_items) > 1 else []
        })
    return items

# %%

text = load_pdf(pdf_path)
structured = split_text(text)

# 1) 조문번호와 제목만 출력
for article in structured:
    print(article['article_number'], article['title'])
print('-' * 40)

# 2) 항(clauses)만 출력
for article in structured:
    for clause in article['clauses']:
        print(clause['clause_text'])
print('-' * 40)

# 3) 호(items)만 출력
for article in structured:
    for clause in article['clauses']:
        for item in clause['items']:
            print(item['item_text'])
print('-' * 40)

# 4) 목(sub_items)만 출력
for article in structured:
    for clause in article['clauses']:
        for item in clause['items']:
            for sub in item['sub_items']:
                print(sub)

# %%

structured[3]