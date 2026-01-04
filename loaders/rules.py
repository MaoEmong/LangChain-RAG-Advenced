"""
loaders/rules.py
============================================================
로더 규칙 정의

파일 확장자별로 사용할 LangChain 로더를 정의합니다.
auto_loader.py에서 이 규칙을 사용하여 적절한 로더를 선택합니다.

주의:
- PDF 파일은 auto_loader.py에서 별도로 처리됩니다
- 이 규칙의 PDF 로더는 실제로는 사용되지 않을 수 있습니다
"""

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
)

def get_loader_rules():
    """
    파일 확장자별 로더 규칙을 반환합니다.
    
    Returns:
        list[tuple]: (확장자, 로더 팩토리 함수) 튜플의 리스트
            - 확장자: 파일 확장자 (예: ".txt", ".md")
            - 로더 팩토리: 파일 경로를 받아 로더를 생성하는 함수
    
    지원 형식:
        - .txt: 텍스트 파일 (UTF-8 인코딩)
        - .md: 마크다운 파일 (UTF-8 인코딩)
        - .pdf: PDF 파일 (PyPDFLoader 사용, 실제로는 auto_loader에서 별도 처리)
        - .docx: Word 문서
        - .html/.htm: HTML 파일
    
    Note:
        - PDF는 auto_loader.py에서 텍스트/스캔 PDF로 구분하여 처리됩니다
        - 이 규칙의 PDF 로더는 참고용이며 실제 사용되지 않을 수 있습니다
    """
    return [
        (".txt",  lambda p: TextLoader(p, encoding="utf-8")),  # 텍스트 파일
        (".md", lambda p: TextLoader(p, encoding="utf-8")),   # 마크다운 파일
        (".pdf",  lambda p: PyPDFLoader(p)),                    # PDF 파일 (참고용)
        (".docx", lambda p: Docx2txtLoader(p)),                 # Word 문서
        (".html", lambda p: BSHTMLLoader(p)),                   # HTML 파일
        (".htm",  lambda p: BSHTMLLoader(p)),                   # HTML 파일 (짧은 확장자)
    ]
