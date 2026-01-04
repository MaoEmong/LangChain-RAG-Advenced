"""
loaders/pdf_scan_loader.py
============================================================
스캔 PDF 로더 (OCR 기반)

텍스트 레이어가 없는 스캔 PDF 파일을 OCR로 처리하여 로드합니다.
Tesseract OCR을 사용하여 이미지에서 텍스트를 추출합니다.

특징:
- OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출
- 파일당 Document 1개로 통합 로드 (페이지별 분할 없음)
- 모든 페이지의 OCR 결과를 하나의 문서로 합침
"""

import os
from langchain_core.documents import Document
from preprocess.text_cleaner import clean_text

def load_pdf_scan(path: str, ocr, zoom: float = 2.5) -> list[Document]:
    """
    스캔 PDF 파일을 OCR로 처리하여 Document 리스트로 반환합니다.
    
    이미지 기반 PDF의 각 페이지를 OCR로 처리하여 텍스트를 추출하고,
    모든 페이지의 텍스트를 하나의 Document로 통합합니다.
    
    Args:
        path: PDF 파일 경로
        ocr: OCR 엔진 객체 (TesseractOCR 등)
        zoom: PDF 렌더링 확대 배율 (기본값: 2.5)
            - 클수록 OCR 정확도 상승, 속도/메모리 사용량 증가
            - 작을수록 빠르지만 정확도 감소
    
    Returns:
        list[Document]: Document 1개를 포함한 리스트
            - page_content: 모든 페이지 OCR 결과가 합쳐진 전체 텍스트
            - metadata:
                - source: 파일 절대 경로
                - page: None (parent 문서이므로 페이지 의미 없음)
                - kind: "pdf_scan_ocr"
                - n_pages: 총 페이지 수
    
    Note:
        - 빈 페이지는 제외됩니다
        - 텍스트는 clean_text()로 정리됩니다
        - zoom 값은 OCR 정확도와 성능 사이의 트레이드오프를 조절합니다
    """
    # OCR로 모든 페이지 처리
    pages = ocr.ocr_pdf(path, zoom=zoom)

    # 각 페이지의 텍스트 추출
    texts = []
    for p in pages:
        t = (p.text or "").strip()
        if t:
            texts.append(t)

    # 모든 페이지 텍스트 합치기
    full_text = "\n\n".join(texts).strip()
    full_text = clean_text(full_text)

    abs_path = os.path.abspath(path)
    return [
        Document(
            page_content=full_text,
            metadata={
                "source": abs_path,
                "page": None,
                "kind": "pdf_scan_ocr",
                "n_pages": len(pages),
            },
        )
    ]
