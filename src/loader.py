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