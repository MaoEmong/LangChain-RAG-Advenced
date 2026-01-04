"""
ocr/tesseract_ocr.py
============================================================
Tesseract OCR 구현

Tesseract OCR 엔진을 사용하여 PDF에서 텍스트를 추출합니다.
PyMuPDF로 PDF를 이미지로 렌더링한 후 Tesseract로 OCR을 수행합니다.

특징:
- PDF 전체 페이지 자동 처리
- 다국어 지원 (한국어, 영어 등)
- 렌더링 품질 조절 가능 (zoom 파라미터)
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


@dataclass
class OCRPage:
    """
    OCR 처리된 페이지 정보
    
    Attributes:
        page: 페이지 번호 (1부터 시작)
        text: 추출된 텍스트
    """
    page: int
    text: str


class TesseractOCR:
    """
    Tesseract OCR 엔진 래퍼 클래스
    
    PDF 파일을 OCR로 처리하여 텍스트를 추출합니다.
    PyMuPDF로 PDF를 이미지로 렌더링한 후 Tesseract OCR로 텍스트를 인식합니다.
    """
    
    def __init__(self, tesseract_cmd: str, lang: str = "eng", psm: int = 6):
        """
        TesseractOCR 초기화
        
        Args:
            tesseract_cmd: Tesseract 실행 파일 경로
            lang: 인식할 언어 코드 (기본값: "eng")
                - 예: "eng", "kor", "eng+kor" (다국어 지원)
            psm: 페이지 세그멘테이션 모드 (기본값: 6)
                - 6: 단일 블록 텍스트로 가정
                - 다른 모드는 Tesseract 문서 참고
        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.psm = psm

    def ocr_pdf(self, pdf_path: str, zoom: float = 2.5) -> List[OCRPage]:
        """
        PDF 전체를 OCR 처리하여 페이지별 결과를 반환합니다.
        
        PDF의 모든 페이지를 이미지로 렌더링한 후 OCR을 수행하여
        각 페이지의 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            zoom: PDF 렌더링 확대 배율 (기본값: 2.5)
                - 클수록 OCR 정확도 상승, 처리 시간/메모리 증가
                - 작을수록 빠르지만 정확도 감소
        
        Returns:
            List[OCRPage]: 페이지별 OCR 결과 리스트
                - 각 OCRPage는 페이지 번호와 추출된 텍스트를 포함
        
        Note:
            - pdf_scan_loader.py와 test_loaders.py에서 공통으로 사용하는 표준 메서드
            - 빈 페이지도 결과에 포함됩니다 (텍스트는 빈 문자열)
        """
        doc = fitz.open(pdf_path)
        results: List[OCRPage] = []

        # PDF 렌더링을 위한 변환 행렬 생성 (확대 배율 적용)
        mat = fitz.Matrix(zoom, zoom)

        # 각 페이지 처리
        for i in range(len(doc)):
            page = doc.load_page(i)
            # PDF 페이지를 이미지로 렌더링 (픽스맵 생성)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # 픽스맵을 PIL Image로 변환
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Tesseract OCR 설정 (페이지 세그멘테이션 모드)
            config = f"--psm {self.psm}"
            # OCR 수행
            text = pytesseract.image_to_string(img, lang=self.lang, config=config) or ""
            results.append(OCRPage(page=i + 1, text=text))

        return results
