# build_index.py
from src.loader import load_pdf, split_text
from src.embeddings import EmbeddingStore

def main():

    pdf_path = "data/국민건강보험법.pdf"

    text = load_pdf(pdf_path)
    chunks = split_text(text)
    # 임베딩 생성 및 ChromaDB 색인
    store = EmbeddingStore(db_path="./db/chroma_db")
    store.index_chunks(chunks)
    print(f"✓ {len(chunks)}개 청크 색인 완료")

if __name__ == "__main__":
    main()