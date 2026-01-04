"""
ocr/base.py
============================================================
OCR 베이스 클래스

OCR 엔진의 공통 인터페이스를 정의하는 추상 베이스 클래스입니다.
모든 OCR 구현체는 이 클래스를 상속받아 구현해야 합니다.
"""

from abc import ABC, abstractmethod

class BaseOCR(ABC):
    """
    OCR 엔진의 베이스 클래스 (추상 클래스)
    
    모든 OCR 구현체는 이 클래스를 상속받아 extract_text 메서드를 구현해야 합니다.
    """
    
    @abstractmethod
    def extract_text(self, image_bytes: bytes) -> str:
        """
        이미지 바이트에서 텍스트를 추출합니다.
        
        Args:
            image_bytes: 이미지 바이트 데이터
        
        Returns:
            str: 추출된 텍스트
        
        Note:
            - 이 메서드는 서브클래스에서 반드시 구현해야 합니다
        """
        pass
