"""
rag_server.py
============================================================
FastAPI 기반 RAG (Retrieval-Augmented Generation) 서버

이 서버는 다음과 같은 기능을 제공합니다:
1. /chat: 문서 기반 질의응답 (RAG)
2. /command: 자연어 명령을 JSON으로 변환
3. /ask: 사용자 의도를 자동 분류하여 적절한 엔드포인트로 라우팅

주요 특징:
- 2단계 검색 파이프라인: 벡터 검색 + FlashRank Re-Ranking
- 벡터 유사도 기반 문서 검색 (ChromaDB)
- 검색 결과 신뢰도 평가 및 Guardrail
- 명령 실행 전 화이트리스트 검증
- Re-Ranking으로 검색 정확도 향상
"""

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from chains.rag_chain import format_docs  # context 포맷터만 재사용

from prompts.command_prompt import COMMAND_PROMPT_TEMPLATE

from services.confidence import calculate_confidence 
from services.vector_store import create_vector_store 
from services.command_parser import parse_command_json 
from services.command_validator import validate_commands 
from services.intent_classifier import classify_intent 

from services.rerank_flashrank import FlashRankReranker
from services.retrieval import retrieve_with_rerank

from config import (
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHAT_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
    TOP_SCORE_MAX,
    MIN_GOOD_HITS,
    GOOD_HIT_SCORE_MAX,
)

# ============================================================
# FastAPI 애플리케이션 초기화
# ============================================================
app = FastAPI()

# ============================================================
# 벡터 DB 로드
# ============================================================
vector_db = create_vector_store(
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)

# ============================================================
# LLM 설정
# ============================================================
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)

# ============================================================
# 프롬프트 템플릿
# ============================================================
rag_prompt = ChatPromptTemplate.from_template(
    """
너는 문서 기반 RAG QA 시스템이다.
아래 CONTEXT에 있는 정보만 사용해 답변해라.
모르면 "문서에서 근거를 찾지 못했습니다."라고 답해라.
답변의 핵심 문장 끝에는 근거로 사용한 DOC 번호를 (DOC 1)처럼 붙여라.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
"""
)

command_prompt = ChatPromptTemplate.from_template(COMMAND_PROMPT_TEMPLATE)

# ============================================================
# Re-Ranking 설정
# ============================================================
# 2단계 검색 파이프라인:
# 1. 벡터 검색으로 넓은 후보 확보 (INITIAL_K개)
# 2. FlashRank Re-Ranker로 관련성 높은 문서 재정렬
# 3. 상위 TOP_K개만 최종 선택
INITIAL_K = 20  # 초기 후보 문서 개수 (넓게 가져올 개수)
reranker = FlashRankReranker(model="ms-marco-MiniLM-L-12-v2")  # 경량 FlashRank 모델

# ============================================================
# 요청 스키마
# ============================================================
class ChatRequest(BaseModel):
    question: str


def _retrieve(req_question: str):
    """
    통합 검색 함수: 2단계 검색 파이프라인 실행
    
    검색 과정:
    1. 벡터 유사도 검색으로 넓은 후보 확보 (INITIAL_K개)
    2. FlashRank Re-Ranker로 관련성 높은 문서 재정렬
    3. 상위 TOP_K개만 선택하여 반환
    
    Args:
        req_question (str): 검색 질문
    
    Returns:
        List[DocumentScore]: (Document, distance_score) 튜플 리스트
            - rerank된 순서로 정렬됨
            - 원본 벡터 검색의 distance score가 보존됨 (Guardrail/Confidence 계산용)
    """
    return retrieve_with_rerank(
        vector_db=vector_db,
        query=req_question,
        initial_k=INITIAL_K,
        top_k=TOP_K,
        reranker=reranker,
    )


def _guard_and_conf(results):
    """
    Guardrail 및 Confidence 계산에 필요한 값들을 추출합니다.
    
    Args:
        results: _retrieve()가 반환한 (Document, distance_score) 리스트
    
    Returns:
        tuple: (top_score, good_hits, confidence)
            - top_score (float): 최상위 문서의 distance score
            - good_hits (int): GOOD_HIT_SCORE_MAX 이하인 "좋은" 문서 개수
            - confidence (dict): 신뢰도 정보 (level, score, details)
    
    Note:
        - results가 비어있으면 (None, None, None) 반환
        - distance score는 낮을수록 유사 (ChromaDB 기준)
    """
    if not results:
        return None, None, None

    # 최상위 문서의 distance score (rerank 후에도 원본 score 사용)
    top_score = float(results[0][1])
    
    # GOOD_HIT_SCORE_MAX 이하인 "좋은" 문서 개수 계산
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)
    
    # 신뢰도 계산
    confidence = calculate_confidence(top_score, good_hits)
    
    return top_score, good_hits, confidence


def _sources_from_results(results):
    """
    검색 결과를 사용자에게 표시할 출처 정보 형식으로 변환합니다.
    
    Args:
        results: _retrieve()가 반환한 (Document, distance_score) 리스트
    
    Returns:
        list: 출처 정보 딕셔너리 리스트
            - source: 문서 출처 경로
            - score: 유사도 점수 (distance)
            - preview: 문서 내용 미리보기 (100자)
    """
    sources = []
    for d, score in results:
        sources.append({
            "source": d.metadata.get("source"),  # 문서 출처 경로
            "score": float(score),               # 유사도 점수
            "preview": d.page_content[:100],     # 문서 내용 미리보기
        })
    return sources


@app.post("/chat")
def chat(req: ChatRequest):
    """
    /chat 엔드포인트: 문서 기반 질의응답 (RAG)
    
    사용자 질문에 대해:
    1. 2단계 검색 파이프라인 실행 (벡터 검색 + FlashRank Re-Ranking)
    2. Guardrail 검사 (검색 결과, 점수, 좋은 문서 개수)
    3. 검색된 문서를 컨텍스트로 사용하여 LLM이 답변 생성
    
    Args:
        req: ChatRequest 객체 (question 필드 포함)
    
    Returns:
        dict: 답변, 출처, 신뢰도 정보를 포함한 응답
    """
    # ============================================================
    # 1단계: 통합 검색 + Re-Ranking
    # ============================================================
    # 벡터 검색으로 넓은 후보 확보 → FlashRank로 재정렬 → 상위 문서 선택
    results = _retrieve(req.question)

    # ============================================================
    # Guardrail 1: 검색 결과 없음
    # ============================================================
    if not results:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 근거를 찾지 못했습니다.",
            "sources": [],
            "guard": {"reason": "no_results"},
        }

    # ============================================================
    # Guardrail 및 Confidence 계산
    # ============================================================
    top_score, good_hits, confidence = _guard_and_conf(results)

    # ============================================================
    # Guardrail 2: TOP1 점수 컷
    # ============================================================
    # 최상위 문서의 유사도가 너무 낮으면 신뢰할 수 없음
    if top_score > TOP_SCORE_MAX:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 충분한 근거를 찾지 못했습니다.",
            "sources": [],
            "guard": {"reason": "low_confidence", "top_score": top_score},
        }

    # ============================================================
    # Guardrail 3: 충분히 좋은 문서 수 부족
    # ============================================================
    # 좋은 품질의 문서가 최소 개수보다 적으면 신뢰도 낮음
    if good_hits < MIN_GOOD_HITS:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 충분한 근거를 찾지 못했습니다.",
            "sources": [],
            "guard": {"reason": "insufficient_good_hits", "good_hits": good_hits},
        }

    # ============================================================
    # ✅ 정상 성공 경로: 모든 Guardrail 통과
    # ============================================================
    # Re-Ranking된 문서로만 컨텍스트 구성 (중요!)
    # rerank 결과가 더 정확하므로 rerank된 문서만 사용
    docs_only = [d for d, _ in results]
    context = format_docs(docs_only)
    
    # 프롬프트에 컨텍스트와 질문 삽입
    messages = rag_prompt.format_messages(context=context, question=req.question)
    
    # LLM이 컨텍스트 기반으로 답변 생성
    answer = llm.invoke(messages).content

    return {
        "type": "rag_answer",
        "question": req.question,
        "answer": answer,
        "sources": _sources_from_results(results),
        "guard": {"reason": "ok", "top_score": top_score, "good_hits": good_hits},
        "confidence": confidence,
    }


@app.post("/command")
def command(req: ChatRequest):
    """
    /command 엔드포인트: 자연어 명령을 JSON 형식의 실행 가능한 명령으로 변환
    
    사용자의 자연어 명령을 분석하여:
    1. 2단계 검색 파이프라인 실행 (벡터 검색 + FlashRank Re-Ranking)
    2. 신뢰도 평가 (낮으면 차단)
    3. Re-Ranking된 문서를 컨텍스트로 사용하여 LLM이 JSON 명령 생성
    4. JSON 파싱 및 검증
    5. 화이트리스트 기반 명령 허용 여부 확인
    
    Args:
        req: ChatRequest 객체 (question 필드 포함)
    
    Returns:
        dict: 명령 타입, 음성 안내, 실행 액션 목록, 신뢰도 정보
    """
    # ============================================================
    # 1단계: 통합 검색 + Re-Ranking
    # ============================================================
    # chat과 동일한 검색 파이프라인 사용 (일관성 유지)
    results = _retrieve(req.question)

    # 검색 결과가 없으면 명령 실행 불가
    if not results:
        return {
            "type": "command",
            "speech": "실행 가능한 명령을 찾지 못했습니다.",
            "actions": [],
            "confidence": {"level": "low", "score": 0.0, "details": {"base": 0.0, "bonus": 0.0}},
            "guard": {"reason": "no_results"},
        }

    # ============================================================
    # 2단계: Guardrail 및 Confidence 계산
    # ============================================================
    top_score, good_hits, confidence = _guard_and_conf(results)

    # ============================================================
    # 3단계: 신뢰도 기반 차단 (명령 실행은 더 엄격한 기준 필요)
    # ============================================================
    # 명령 실행은 더 보수적으로 차단 (안전성 확보)
    if confidence["level"] == "low" and confidence["score"] < 0.5:
        return {
            "type": "command",
            "speech": "확신이 부족하여 명령을 실행할 수 없습니다.",
            "actions": [],
            "confidence": confidence,
            "guard": {"reason": "low_confidence", "top_score": top_score, "good_hits": good_hits},
        }

    # ============================================================
    # 4단계: LLM을 통한 명령 JSON 생성
    # ============================================================
    # Re-Ranking된 문서로만 컨텍스트 구성 (중요!)
    # rerank 결과가 더 정확하므로 rerank된 문서만 사용
    docs_only = [d for d, _ in results]
    context = format_docs(docs_only)

    # 프롬프트에 컨텍스트와 질문 삽입
    messages = command_prompt.format_messages(context=context, question=req.question)
    
    # LLM이 JSON 형식의 명령 생성
    raw_text = llm.invoke(messages).content

    # ============================================================
    # 5단계: JSON 파싱 및 스키마 검증
    # ============================================================
    # LLM이 생성한 JSON 문자열을 CommandResponse 객체로 변환
    parsed = parse_command_json(raw_text)
    
    # 파싱 실패 시 (JSON 형식 오류 또는 스키마 불일치)
    if not parsed:
        return {
            "type": "command",
            "speech": "명령을 해석하지 못했습니다.",
            "actions": [],
            "confidence": confidence,
            "guard": {"reason": "parse_failed"},
            "raw": raw_text,  # 디버깅용 (필요 없으면 제거 가능)
        }

    # ============================================================
    # 6단계: 화이트리스트 기반 명령 검증
    # ============================================================
    # 허용된 명령 목록에 있는지, 필요한 인자가 모두 있는지 확인
    ok, reason = validate_commands(parsed)
    
    # 허용되지 않은 명령이면 차단
    if not ok:
        return {
            "type": "command",
            "speech": "허용되지 않은 명령입니다.",
            "actions": [],
            "confidence": confidence,
            "guard": {"reason": "command_not_allowed", "detail": reason},
        }

    # ============================================================
    # 7단계: 최종 안전한 명령 반환
    # ============================================================
    # 모든 검증을 통과한 명령을 반환
    return {
        "type": "command",
        "speech": parsed.speech,  # 사용자에게 보여줄/말해줄 안내 문구
        "actions": [a.model_dump() for a in parsed.actions],  # 실행할 액션 목록
        "confidence": confidence,
        "sources": _sources_from_results(results),  # 출처 정보 (필요 없으면 제거 가능)
    }


@app.post("/ask")
def ask(req: ChatRequest):
    """
    /ask 엔드포인트: 사용자 의도 자동 분류 및 라우팅
    
    사용자의 입력을 분석하여:
    - "command": 실행/조작 요청 → /command 엔드포인트로 라우팅
    - "explain": 설명/질문 요청 → /chat 엔드포인트로 라우팅
    
    의도 분류는 Rule 기반 + LLM 기반 하이브리드 방식 사용:
    1. Rule 기반: 빠른 패턴 매칭 (해줘, 설명해줘 등)
    2. LLM 기반: 애매한 경우 LLM이 분류
    
    Args:
        req: ChatRequest 객체 (question 필드 포함)
    
    Returns:
        dict: 분류된 의도에 따라 /chat 또는 /command의 응답 반환
    """
    # 사용자 입력의 의도 분류 (command 또는 explain)
    intent = classify_intent(req.question, llm)
    
    # 명령 실행 요청인 경우
    if intent.intent == "command":
        # /command 엔드포인트 로직 재사용
        return command(req)
    
    # 설명/질문 요청인 경우
    # explain이면 기존 /chat 로직 사용
    return chat(req)
