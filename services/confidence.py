"""
services/confidence.py
============================================================
검색 결과 신뢰도 계산

이 모듈은 벡터 검색 결과의 신뢰도를 계산합니다.
신뢰도는 다음 요소들을 종합하여 계산됩니다:
1. 최상위 문서의 유사도 점수
2. 좋은 품질의 문서 개수
"""

from config import CONF_SCORE_MIN, CONF_SCORE_MAX

def normalize_score(score: float) -> float:
    """
    유사도 점수를 0.0 ~ 1.0 범위로 정규화합니다.
    
    ChromaDB의 점수는 distance 기반이므로:
    - 낮을수록 유사 (좋음) → 높은 신뢰도
    - 높을수록 불유사 (나쁨) → 낮은 신뢰도
    
    Args:
        score (float): 유사도 점수 (distance)
    
    Returns:
        float: 정규화된 신뢰도 점수 (0.0 ~ 1.0)
    """
    # 매우 좋은 점수 이하면 최대 신뢰도
    if score <= CONF_SCORE_MIN:
        return 1.0
    
    # 최악의 점수 이상이면 최소 신뢰도
    if score >= CONF_SCORE_MAX:
        return 0.0
    
    # 선형 보간: 점수가 낮을수록 신뢰도 높음
    return 1.0 - (score - CONF_SCORE_MIN) / (CONF_SCORE_MAX - CONF_SCORE_MIN)

def hits_bonus(good_hits: int) -> float:
    """
    좋은 품질의 문서 개수에 따른 보너스 점수를 계산합니다.
    
    여러 개의 좋은 문서가 있으면 신뢰도가 높아집니다.
    이는 검색 결과의 일관성과 신뢰성을 나타냅니다.
    
    Args:
        good_hits (int): GOOD_HIT_SCORE_MAX 이하인 문서 개수
    
    Returns:
        float: 보너스 점수 (0.0 ~ 0.15)
    """
    if good_hits >= 3:
        return 0.15  # 3개 이상이면 최대 보너스
    if good_hits == 2:
        return 0.10  # 2개면 중간 보너스
    if good_hits == 1:
        return 0.05  # 1개면 작은 보너스
    return 0.0       # 없으면 보너스 없음

def calculate_confidence(top_score: float, good_hits: int) -> dict:
    """
    검색 결과의 전체 신뢰도를 계산합니다.
    
    신뢰도는 다음 공식으로 계산됩니다:
    - 기본 신뢰도: 최상위 문서의 점수 기반
    - 보너스: 좋은 문서 개수 기반
    - 최종 신뢰도 = min(기본 + 보너스, 1.0)
    
    Args:
        top_score (float): 최상위 문서의 유사도 점수
        good_hits (int): 좋은 품질의 문서 개수
    
    Returns:
        dict: 신뢰도 정보를 포함한 딕셔너리
            - level: "high" | "medium" | "low"
            - score: 신뢰도 점수 (0.0 ~ 1.0)
            - details: 상세 정보 (base, bonus)
    """
    # 기본 신뢰도: 최상위 문서 점수 기반
    base = normalize_score(top_score)
    
    # 보너스: 좋은 문서 개수 기반
    bonus = hits_bonus(good_hits)
    
    # 최종 신뢰도 (1.0을 넘지 않도록 제한)
    final = min(base + bonus, 1.0)

    # 신뢰도 레벨 분류
    if final >= 0.75:
        level = "high"
    elif final >= 0.5:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": round(final, 3),
        "details": {
            "base": round(base, 3),      # 기본 신뢰도
            "bonus": round(bonus, 3),    # 보너스 점수
        },
    }
