"""
loaders/pdf_detector.py
============================================================
PDF 종류 판별기

PDF 파일이 텍스트 PDF인지 스캔 PDF인지 자동으로 판별합니다.
텍스트 레이어가 있는 PDF는 텍스트 PDF로, 없는 PDF는 스캔 PDF로 판별합니다.

판별 방법:
- PDF의 처음 N페이지에서 텍스트를 추출
- 추출된 텍스트 길이가 최소 기준 이상이면 텍스트 PDF로 판별
- 그렇지 않으면 스캔 PDF로 판별
"""

import fitz  # PyMuPDF

def is_text_pdf(path: str, check_pages: int = 2, min_chars: int = 80) -> bool:
    """
    PDF 파일이 텍스트 PDF인지 스캔 PDF인지 판별합니다.
    
    PDF의 처음 N페이지에서 텍스트를 추출하여 텍스트 레이어 존재 여부를 확인합니다.
    텍스트 레이어가 있으면 텍스트 PDF, 없으면 스캔 PDF로 판별합니다.
    
    Args:
        path: PDF 파일 경로
        check_pages: 확인할 페이지 수 (기본값: 2페이지)
        min_chars: 텍스트 PDF로 판별하기 위한 최소 문자 수 (기본값: 80자)
    
    Returns:
        bool: 텍스트 PDF이면 True, 스캔 PDF이면 False
    
    Note:
        - 처음 몇 페이지만 확인하므로 빠른 판별이 가능합니다
        - min_chars 미만의 텍스트가 추출되면 스캔 PDF로 판별됩니다
    """
    doc = fitz.open(path)
    pages = min(check_pages, doc.page_count)
    total = 0

    # 처음 N페이지에서 텍스트 추출
    for i in range(pages):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        total += len(text)

    doc.close()
    
    # 최소 문자 수 이상이면 텍스트 PDF로 판별
    return total >= min_chars
