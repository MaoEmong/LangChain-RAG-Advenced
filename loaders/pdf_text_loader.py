"""
loaders/pdf_text_loader.py
============================================================
텍스트 PDF 로더

텍스트 레이어가 있는 PDF 파일을 로드합니다.
PyMuPDF(fitz)를 사용하여 텍스트를 직접 추출합니다.

특징:
- 파일당 Document 1개로 통합 로드 (페이지별 분할 없음)
- 모든 페이지의 텍스트를 하나의 문서로 합침
- 텍스트 정리 후 반환
"""

import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from preprocess.text_cleaner import clean_text

def load_pdf_text(path: str) -> list[Document]:
    """
    텍스트 PDF 파일을 로드하여 Document 리스트로 반환합니다.
    
    텍스트 레이어가 있는 PDF에서 모든 페이지의 텍스트를 추출하여
    하나의 Document로 통합합니다. 페이지별로 분할하지 않습니다.
    
    Args:
        path: PDF 파일 경로
    
    Returns:
        list[Document]: Document 1개를 포함한 리스트
            - page_content: 모든 페이지 텍스트가 합쳐진 전체 텍스트
            - metadata:
                - source: 파일 절대 경로
                - page: None (parent 문서이므로 페이지 의미 없음)
                - kind: "pdf_text"
                - n_pages: 총 페이지 수 (참고용)
    
    Note:
        - 빈 페이지는 제외됩니다
        - 텍스트는 clean_text()로 정리됩니다
    """
    doc = fitz.open(path)
    pages = []
    
    # 모든 페이지에서 텍스트 추출
    for pno in range(len(doc)):
        text = doc.load_page(pno).get_text("text") or ""
        text = text.strip()
        if text:
            pages.append(text)

    # 모든 페이지 텍스트 합치기
    full_text = "\n\n".join(pages).strip()
    full_text = clean_text(full_text)

    abs_path = os.path.abspath(path)
    return [
        Document(
            page_content=full_text,
            metadata={
                "source": abs_path,
                "page": None,          # parent 문서이므로 페이지 의미 없음
                "kind": "pdf_text",
                "n_pages": len(doc),   # 참고용 페이지 수
            },
        )
    ]
