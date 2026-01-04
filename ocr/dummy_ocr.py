"""
ocr/dummy_ocr.py
============================================================
더미 OCR 구현

OCR 기능이 아직 연결되지 않았을 때 사용하는 더미 OCR 클래스입니다.
테스트나 개발 단계에서 사용할 수 있습니다.

참고:
- 실제 OCR 기능이 필요한 경우 TesseractOCR을 사용하세요
- 이 클래스는 항상 빈 문자열을 반환합니다
"""

from ocr.base import BaseOCR

class DummyOCR(BaseOCR):
    """
    더미 OCR 구현 클래스
    
    OCR 기능이 연결되기 전의 임시 구현체입니다.
    모든 이미지에 대해 빈 문자열을 반환합니다.
    """
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        더미 텍스트 추출 (항상 빈 문자열 반환)
        
        Args:
            image_bytes: 이미지 바이트 데이터 (사용하지 않음)
        
        Returns:
            str: 항상 빈 문자열 ""
        
        Note:
            - 실제 OCR이 필요할 때는 TesseractOCR을 사용하세요
        """
        return ""
