"""
preprocess/text_cleaner.py
============================================================
텍스트 정리 유틸리티

추출된 텍스트를 정리하여 일관된 형식으로 만듭니다.
개행 문자 통일, 과도한 공백 제거 등을 수행합니다.
"""

import re

def clean_text(text: str) -> str:
    """
    텍스트를 정리하여 일관된 형식으로 변환합니다.
    
    수행하는 작업:
    1. 개행 문자 통일 (Windows/Mac/Linux 개행 → Unix 스타일)
    2. 과도한 빈 줄 제거 (3줄 이상 → 2줄)
    3. 과도한 공백/탭 제거 (2개 이상 → 1개)
    4. 앞뒤 공백 제거
    
    Args:
        text: 정리할 텍스트
    
    Returns:
        str: 정리된 텍스트
    
    Note:
        - 빈 문자열이나 None 입력 시 빈 문자열 반환
        - 텍스트 정리 후 앞뒤 공백은 제거됩니다
    """
    if not text:
        return ""
    
    # 1) 개행 문자 통일 (Windows/Mac → Unix 스타일)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) 과도한 공백/빈줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3줄 이상 빈 줄 → 2줄
    text = re.sub(r"[ \t]{2,}", " ", text)  # 2개 이상 공백/탭 → 1개

    return text.strip()
