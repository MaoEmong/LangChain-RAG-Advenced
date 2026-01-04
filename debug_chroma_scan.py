"""
debug_chroma_scan.py
============================================================
ChromaDB 스캔 문서 디버깅 스크립트

ChromaDB에 저장된 스캔 PDF 문서(OCR 결과)를 확인하는 디버깅 스크립트입니다.

기능:
- ChromaDB에서 스캔 PDF 문서 직접 조회
- 메타데이터 및 문서 내용 확인
"""

from ingest_langchain import build_or_load_chroma

def main():
    """
    ChromaDB에서 스캔 PDF 문서를 조회하여 출력합니다.
    """
    db = build_or_load_chroma()

    scan_src = r"D:\langchain-rag-advanced\docs\Scanned_20260102-1507.pdf"

    # ChromaDB 내부 raw collection 접근
    col = db._collection

    # kind 메타데이터로 필터 조회 (children chunks가 들어있어야 함)
    res = col.get(
        where={"kind": "pdf_scan_ocr"},
        include=["metadatas", "documents"]
    )

    print("=== CHROMA GET BY SOURCE ===")
    print("ids:", len(res.get("ids", [])))
    
    # 메타데이터 샘플 출력
    if res.get("metadatas"):
        print("\n[SAMPLE METADATA]")
        print(res["metadatas"][0])
    
    # 문서 내용 샘플 출력 (상위 3개)
    if res.get("documents"):
        for i, txt in enumerate(res["documents"][:3], 1):
            print(f"\n[{i}] chars={len(txt)}")
            print(txt[:300])

if __name__ == "__main__":
    main()
