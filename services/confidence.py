from config import CONF_SCORE_MIN, CONF_SCORE_MAX

def normalize_score(score: float) -> float:
    if score <= CONF_SCORE_MIN:
        return 1.0
    if score >= CONF_SCORE_MAX:
        return 0.0
    return 1.0 - (score - CONF_SCORE_MIN) / (CONF_SCORE_MAX - CONF_SCORE_MIN)

def hits_bonus(good_hits: int) -> float:
    if good_hits >= 3:
        return 0.15
    if good_hits == 2:
        return 0.10
    if good_hits == 1:
        return 0.05
    return 0.0

def calculate_confidence(top_score: float, good_hits: int) -> dict:
    base = normalize_score(top_score)
    bonus = hits_bonus(good_hits)
    final = min(base + bonus, 1.0)

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
            "base": round(base, 3),
            "bonus": round(bonus, 3),
        },
    }
