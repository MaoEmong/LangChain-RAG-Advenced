"""
rag_server.py
============================================================
FastAPI 기반 RAG (Retrieval-Augmented Generation) 서버

이 서버는 다음과 같은 기능을 제공합니다:
1. /chat: 문서 기반 질의응답 (RAG)
2. /command: 자연어 명령을 JSON으로 변환
3. /ask: 사용자 의도를 자동 분류하여 적절한 엔드포인트로 라우팅

주요 특징:
- 벡터 유사도 기반 문서 검색 (ChromaDB)
- 검색 결과 신뢰도 평가 및 Guardrail
- MMR (Maximal Marginal Relevance) 기반 검색으로 다양성 확보
- 명령 실행 전 화이트리스트 검증
"""

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from chains.rag_chain import build_rag_chain
from chains.command_chain import build_command_chain

from prompts.command_prompt import COMMAND_PROMPT_TEMPLATE

from services.confidence import calculate_confidence
from services.vector_store import create_vector_store
from services.command_parser import parse_command_json
from services.command_validator import validate_commands
from services.intent_classifier import classify_intent

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
# 벡터 데이터베이스 및 검색기 설정
# ============================================================
# ChromaDB 벡터 저장소 생성 (이미 저장된 벡터DB 로드)
vector_db = create_vector_store(
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)

# MMR (Maximal Marginal Relevance) 기반 검색기 설정
# - search_type="mmr": 유사도와 다양성을 모두 고려한 검색
# - k: 최종 반환할 문서 개수
# - fetch_k: 다양성을 위해 먼저 가져올 문서 개수 (k의 3배, 최소 12개)
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": max(12, TOP_K * 3),
    },
)

# ============================================================
# LLM (Large Language Model) 설정
# ============================================================
# OpenAI Chat 모델 초기화
# - temperature=0.2: 낮은 값으로 일관된 답변 생성
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)

# ============================================================
# 프롬프트 템플릿 설정
# ============================================================
# RAG 질의응답용 프롬프트
# - CONTEXT: 검색된 문서들
# - QUESTION: 사용자 질문
# - ANSWER: LLM이 생성할 답변
prompt = ChatPromptTemplate.from_template(
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

# 명령 생성용 프롬프트 (prompts/command_prompt.py에서 가져옴)
command_prompt = ChatPromptTemplate.from_template(COMMAND_PROMPT_TEMPLATE)

# ============================================================
# LangChain 체인 구성
# ============================================================
# 명령 생성 체인: 사용자 입력 -> 문서 검색 -> JSON 명령 생성
command_chain = build_command_chain(retriever, llm, command_prompt)

# RAG 체인: 사용자 질문 -> 문서 검색 -> 컨텍스트 구성 -> 답변 생성
rag_chain = build_rag_chain(retriever, llm, prompt)

# ============================================================
# 요청/응답 스키마 정의
# ============================================================
class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    question: str  # 사용자 질문

@app.post("/chat")
def chat(req: ChatRequest):
    """
    /chat 엔드포인트: 문서 기반 질의응답 (RAG)
    
    사용자 질문에 대해 벡터DB에서 관련 문서를 검색하고,
    검색된 문서를 컨텍스트로 사용하여 LLM이 답변을 생성합니다.
    
    Guardrail 체크:
    1. 검색 결과가 없는 경우
    2. 최상위 문서의 유사도 점수가 너무 낮은 경우
    3. 충분히 좋은 품질의 문서가 부족한 경우
    
    Args:
        req: ChatRequest 객체 (question 필드 포함)
    
    Returns:
        dict: 답변, 출처, 신뢰도 정보를 포함한 응답
    """
    # 벡터 유사도 검색 (점수 포함)
    # similarity_search_with_score: (Document, score) 튜플 리스트 반환
    results = vector_db.similarity_search_with_score(req.question, k=TOP_K)

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

    # 최상위 문서의 유사도 점수 (낮을수록 유사)
    top_score = float(results[0][1])
    
    # GOOD_HIT_SCORE_MAX 이하인 "좋은" 문서 개수 계산
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)

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
            "guard": {
                "reason": "low_confidence",
                "top_score": top_score,
            },
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
            "guard": {
                "reason": "insufficient_good_hits",
                "good_hits": good_hits,
            },
        }

    # ============================================================
    # ✅ 정상 성공 경로: 모든 Guardrail 통과
    # ============================================================

    # RAG 체인을 통해 답변 생성
    # 1. 질문으로 문서 검색
    # 2. 검색된 문서를 컨텍스트로 구성
    # 3. LLM이 컨텍스트 기반으로 답변 생성
    answer = rag_chain.invoke(req.question)
    
    # 검색 결과의 신뢰도 계산
    confidence = calculate_confidence(top_score, good_hits)
    
    # 출처 정보 구성 (사용자에게 표시할 용도)
    sources = []
    for d, score in results:
        sources.append({
            "source": d.metadata.get("source"),  # 문서 출처 경로
            "score": float(score),               # 유사도 점수
            "preview": d.page_content[:200],     # 문서 내용 미리보기 (200자)
        })

    return {
        "type": "rag_answer",
        "question": req.question,
        "answer": answer,
        "sources": sources,
        "guard": {
            "reason": "ok",
            "top_score": top_score,
            "good_hits": good_hits,
        },
        "confidence": confidence,
    }

@app.post("/command")
def command(req: ChatRequest):
    """
    /command 엔드포인트: 자연어 명령을 JSON 형식의 실행 가능한 명령으로 변환
    
    사용자의 자연어 명령을 분석하여:
    1. 문서에서 관련 함수 정보 검색
    2. 신뢰도 평가 (낮으면 차단)
    3. LLM이 JSON 형식의 명령 생성
    4. JSON 파싱 및 검증
    5. 화이트리스트 기반 명령 허용 여부 확인
    
    Args:
        req: ChatRequest 객체 (question 필드 포함)
    
    Returns:
        dict: 명령 타입, 음성 안내, 실행 액션 목록, 신뢰도 정보
    """
    # ============================================================
    # 1단계: 벡터 검색 및 신뢰도 계산
    # ============================================================
    # 문서에서 관련 함수/명령 정보 검색
    results = vector_db.similarity_search_with_score(req.question, k=TOP_K)

    # 검색 결과가 없으면 명령 실행 불가
    if not results:
        return {
            "type": "command",
            "speech": "실행 가능한 명령을 찾지 못했습니다.",
            "actions": [],
            "confidence": {"level": "low"},
        }

    # 검색 결과 분석
    top_score = float(results[0][1])
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)
    confidence = calculate_confidence(top_score, good_hits)
    
    # ============================================================
    # 2단계: 신뢰도 기반 차단 (명령 실행은 더 엄격한 기준 필요)
    # ============================================================
    COMMAND_HIGH_THRESHOLD = 0.65  # 명령 실행을 위한 최소 신뢰도
    
    # 신뢰도가 낮으면 명령 실행 차단 (안전성 확보)
    if confidence["level"] == "low" and confidence["score"] < 0.5:
        return {
            "type": "command",
            "speech": "확신이 부족하여 명령을 실행할 수 없습니다.",
            "actions": [],
            "confidence": confidence,
        }

    # ============================================================
    # 3단계: LLM을 통한 명령 JSON 생성
    # ============================================================
    # command_chain: 사용자 입력 -> 문서 검색 -> JSON 명령 생성
    raw_text = command_chain.invoke(req.question)

    # ============================================================
    # 4단계: JSON 파싱 및 스키마 검증
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
        }

    # ============================================================
    # 5단계: 화이트리스트 기반 명령 검증
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
            "guard": {
                "reason": "command_not_allowed",
                "detail": reason,
            },
        }

    # ============================================================
    # 6단계: 최종 안전한 명령 반환
    # ============================================================
    # 모든 검증을 통과한 명령을 반환
    return {
        "type": "command",
        "speech": parsed.speech,  # 사용자에게 보여줄/말해줄 안내 문구
        "actions": [a.model_dump() for a in parsed.actions],  # 실행할 액션 목록
        "confidence": confidence,
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