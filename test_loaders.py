"""
test_loaders.py
============================================================
로더 테스트 스크립트

문서 로더 기능을 테스트하는 스크립트입니다.

테스트 내용:
1. PDF 종류 판별 테스트 (텍스트 PDF vs 스캔 PDF)
2. 텍스트 PDF 추출 샘플 출력
3. AutoLoader 전체 요약
4. 스캔 PDF OCR 샘플 출력

실행:
    python test_loaders.py
"""

import os
from collections import Counter

from loaders.pdf_detector import is_text_pdf
from loaders.pdf_text_loader import load_pdf_text
from loaders.auto_loader import load_docs_from_folder

DOCS_DIR = "./docs"

def preview_doc(d, n=250):
    """
    문서 내용을 미리보기 형식으로 반환합니다.
    
    Args:
        d: Document 객체
        n: 미리보기 문자 수 (기본값: 250)
    
    Returns:
        str: 미리보기 문자열 (길이 초과 시 "..." 추가)
    """
    text = (d.page_content or "").strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")

def test_pdf_detector():
    """
    PDF 종류 판별 테스트
    
    docs 폴더의 모든 PDF 파일을 검사하여
    텍스트 PDF인지 스캔 PDF인지 판별합니다.
    """
    pdfs = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    if not pdfs:
        print("[WARN] docs 폴더에 PDF가 없어. docs/에 pdf 넣고 다시 실행해줘.")
        return

    print("\n===== 1) PDF 판별 테스트 (텍스트PDF vs 스캔PDF) =====")
    for p in pdfs:
        try:
            flag = is_text_pdf(p)
            print(f"- {os.path.basename(p)} => {'TEXT_PDF' if flag else 'SCAN_PDF'}")
        except Exception as e:
            print(f"- {os.path.basename(p)} => ERROR: {e}")

def test_pdf_text_extract_one():
    """
    텍스트 PDF 추출 샘플 테스트
    
    docs 폴더에서 첫 번째 텍스트 PDF를 찾아
    추출된 텍스트의 샘플을 출력합니다.
    """
    # docs에서 첫 번째 텍스트 PDF 하나 골라서 샘플 출력
    pdfs = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))

    for p in pdfs:
        if is_text_pdf(p):
            print("\n===== 2) 텍스트 PDF 추출 샘플 출력 =====")
            docs = load_pdf_text(p)
            print(f"[FILE] {p}")
            print(f"[PAGES] {len(docs)}")
            # 앞 3페이지 미리보기
            for i, d in enumerate(docs[:3], start=1):
                print(f"\n--- page {d.metadata.get('page')} (chars={len(d.page_content)}) ---")
                print(preview_doc(d))
            return

    print("\n[WARN] 텍스트 PDF로 판별된 파일이 없음. (모두 스캔 PDF일 수 있어)")

def test_auto_loader_summary():
    """
    AutoLoader 전체 요약 테스트
    
    AutoLoader로 로드된 모든 문서의 통계를 출력합니다.
    - 문서 종류별 개수
    - 빈 문서 개수
    - 평균 텍스트 길이
    - 샘플 문서 정보
    """
    print("\n===== 3) AutoLoader 전체 요약 (kind/빈페이지/평균길이) =====")
    docs = load_docs_from_folder(DOCS_DIR)
    if not docs:
        print("[WARN] 로딩된 문서가 없음.")
        return

    kind_counts = Counter(d.metadata.get("kind", "?") for d in docs)

    lengths = [len((d.page_content or "").strip()) for d in docs]
    empty = sum(1 for x in lengths if x == 0)
    avg_len = sum(lengths) / len(lengths)

    print(f"[TOTAL DOCS] {len(docs)}")
    print(f"[KINDS] {dict(kind_counts)}")
    print(f"[EMPTY DOCS] {empty} / {len(docs)}")
    print(f"[AVG CHARS] {avg_len:.1f}")

    # 상위 5개만 샘플 출력
    print("\n[SAMPLES] (top 5)")
    for d in docs[:5]:
        print(f"- kind={d.metadata.get('kind')} source={os.path.basename(d.metadata.get('source',''))} page={d.metadata.get('page')}")
        print(f"  {preview_doc(d)}")

    by_source = {}
    for d in docs:
        s = d.metadata.get("source")
        by_source.setdefault(s, set()).add(d.metadata.get("doc_id"))
    for s, ids in list(by_source.items())[:5]:
        print(os.path.basename(s), "doc_id_count =", len(ids))

def test_scan_pdf_preview():
    """
    스캔 PDF OCR 샘플 테스트
    
    특정 스캔 PDF 파일을 OCR로 처리하여
    추출된 텍스트의 샘플을 출력합니다.
    """
    import os
    from loaders.pdf_scan_loader import load_pdf_scan
    from ocr.tesseract_ocr import TesseractOCR

    scan = "./docs/Scanned_20260102-1507.pdf"
    if not os.path.exists(scan):
        print("[WARN] scan pdf not found:", scan)
        return

    # Tesseract OCR 초기화 (한국어+영어 지원)
    ocr = TesseractOCR(
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        lang="eng+kor"
    )

    # 스캔 PDF OCR 처리
    docs = load_pdf_scan(scan, ocr=ocr, zoom=2.5)

    print("\n===== 2-EX) 스캔 PDF OCR 샘플 출력 =====")
    for d in docs[:2]:
        text = (d.page_content or "").strip()
        print(f"--- page {d.metadata.get('page')} (chars={len(text)}) ---")
        print(text[:300])

def main():
    test_pdf_detector()
    test_pdf_text_extract_one()
    test_auto_loader_summary()
    test_scan_pdf_preview()

if __name__ == "__main__":
    main()
