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
- ParentDocument 방식: child(청크)로 검색하고 parent(원문)을 복원해서 LLM에 제공
- 벡터 유사도 기반 문서 검색 (ChromaDB)
- 검색 결과 신뢰도 평가 및 Guardrail
- 명령 실행 전 화이트리스트 검증
- Re-Ranking으로 검색 정확도 향상
"""

from fastapi import FastAPI
from pydantic import BaseModel

from typing import List, Tuple
from langchain_core.documents import Document

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
from services.retrieval import retrieve_parents_with_rerank

from docstore_sqlite import SQLiteDocStore

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
    DOCSTORE_PATH,
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

# Parent DocStore (sqlite)
docstore = SQLiteDocStore(DOCSTORE_PATH)

DocumentScore = Tuple[Document, float]  # (Document, distance_score)

# ============================================================
# LLM 설정
# ============================================================
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)

# ============================================================
# 컨텍스트 길이 제한 (중요)
# - parent 문서가 길어지면 LLM이 회피 답변을 할 확률이 늘어남
# - 여기서 서버가 강제로 잘라서 안정화
# ============================================================
MAX_CONTEXT_CHARS = 12000  # 필요하면 8000~20000 사이로 튜닝

def _trim_context(context: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(context) <= limit:
        return context
    # 너무 딱 자르면 DOC 블록이 중간에서 끊길 수 있으니, 마지막 DOC 경계에서 자르려 시도
    cut = context.rfind("\n\n[DOC", 0, limit)
    if cut == -1 or cut < limit * 0.5:
        # 경계 찾기 실패하면 그냥 limit에서 자름
        return context[:limit]
    return context[:cut].rstrip()

# ============================================================
# 프롬프트 템플릿
# ============================================================
# ✅ 핵심 수정:
# - "모르면 그 문장" 규칙은 유지하되
# - CONTEXT에 프로그램 소개/개요가 있으면 반드시 그걸로 답하도록 더 강하게 지시
rag_prompt = ChatPromptTemplate.from_template(
    """
너는 문서 기반 RAG QA 시스템이다.
반드시 아래 CONTEXT에 있는 정보만 사용해서 답해라.

규칙:
1) CONTEXT에 질문과 관련된 정보가 있으면, 그 내용을 요약해서 답해야 한다.
2) CONTEXT에 정말 아무 근거가 없을 때만 "문서에서 근거를 찾지 못했습니다."라고 답한다.
3) 답변의 핵심 문장 끝에는 근거로 사용한 DOC 번호를 (DOC 1)처럼 붙여라.
4) 과장/추측/상상 금지. 문서에 있는 표현을 우선 사용하되, 자연스럽게 풀어서 설명해라.

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
INITIAL_K = 20
reranker = FlashRankReranker(model="ms-marco-MiniLM-L-12-v2")

# ============================================================
# 요청 스키마
# ============================================================
class ChatRequest(BaseModel):
    question: str

def _retrieve(req_question: str) -> List[DocumentScore]:
    """
    Parent 기반 검색:
    1) child(청크) 후보를 벡터검색으로 넓게 가져옴
    2) rerank
    3) parent(docstore) 복원
    """
    return retrieve_parents_with_rerank(
        vector_db=vector_db,
        docstore=docstore,
        query=req_question,
        initial_k=INITIAL_K,
        top_k=TOP_K,
        reranker=reranker,
        parent_id_key="doc_id",
    )

def _guard_and_conf(results: List[DocumentScore]):
    """검색 결과를 기반으로 guardrail 판단에 필요한 요약 지표를 계산."""
    if not results:
        return None, None, None

    top_score = float(results[0][1])
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)
    confidence = calculate_confidence(top_score, good_hits)
    return top_score, good_hits, confidence

def _sources_from_results(results: List[DocumentScore]):
    """프론트/로그에서 확인할 수 있도록 간략한 source 정보만 추출."""
    sources = []
    for d, score in results:
        sources.append({
            "source": d.metadata.get("source"),
            "score": float(score),
            "preview": d.page_content[:180],
        })
    return sources

# ============================================================
# /chat
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest):
    results = _retrieve(req.question)

    if not results:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 근거를 찾지 못했습니다.",
            "sources": [],
            "guard": {"reason": "no_results"},
        }

    top_score, good_hits, confidence = _guard_and_conf(results)

    # guard 실패여도 sources는 같이 내려서 디버깅/UX 개선
    if top_score is not None and top_score > TOP_SCORE_MAX:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 충분한 근거를 찾지 못했습니다.",
            "sources": _sources_from_results(results),
            "guard": {"reason": "low_confidence", "top_score": top_score, "good_hits": good_hits},
            "confidence": confidence,
        }

    has_parent_context = any(len(d.page_content) > 300 for d, _ in results)
    
    if good_hits is not None and good_hits < MIN_GOOD_HITS and not has_parent_context:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 충분한 근거를 찾지 못했습니다.",
            "sources": _sources_from_results(results),
            "guard": {
                "reason": "insufficient_good_hits",
                "top_score": top_score,
                "good_hits": good_hits,
            },
            "confidence": confidence,
        }

    docs_only = [d for d, _ in results]
    context = format_docs(docs_only)
    context = _trim_context(context)

    messages = rag_prompt.format_messages(context=context, question=req.question)
    answer = llm.invoke(messages).content

    return {
        "type": "rag_answer",
        "question": req.question,
        "answer": answer,
        "sources": _sources_from_results(results),
        "guard": {"reason": "ok", "top_score": top_score, "good_hits": good_hits},
        "confidence": confidence,
    }

# ============================================================
# /command
# ============================================================
@app.post("/command")
def command(req: ChatRequest):
    results = _retrieve(req.question)

    if not results:
        return {
            "type": "command",
            "speech": "실행 가능한 명령을 찾지 못했습니다.",
            "actions": [],
            "confidence": {"level": "low", "score": 0.0, "details": {"base": 0.0, "bonus": 0.0}},
            "guard": {"reason": "no_results"},
        }

    top_score, good_hits, confidence = _guard_and_conf(results)

    # 명령은 보수적으로
    if confidence["level"] == "low" and confidence["score"] < 0.5:
        return {
            "type": "command",
            "speech": "확신이 부족하여 명령을 실행할 수 없습니다.",
            "actions": [],
            "confidence": confidence,
            "sources": _sources_from_results(results),
            "guard": {"reason": "low_confidence", "top_score": top_score, "good_hits": good_hits},
        }

    docs_only = [d for d, _ in results]
    context = format_docs(docs_only)
    context = _trim_context(context)

    messages = command_prompt.format_messages(context=context, question=req.question)
    raw_text = llm.invoke(messages).content

    parsed = parse_command_json(raw_text)
    if not parsed:
        return {
            "type": "command",
            "speech": "명령을 해석하지 못했습니다.",
            "actions": [],
            "confidence": confidence,
            "sources": _sources_from_results(results),
            "guard": {"reason": "parse_failed"},
            "raw": raw_text,
        }

    ok, reason = validate_commands(parsed)
    if not ok:
        return {
            "type": "command",
            "speech": "허용되지 않은 명령입니다.",
            "actions": [],
            "confidence": confidence,
            "sources": _sources_from_results(results),
            "guard": {"reason": "command_not_allowed", "detail": reason},
        }

    return {
        "type": "command",
        "speech": parsed.speech,
        "actions": [a.model_dump() for a in parsed.actions],
        "confidence": confidence,
        "sources": _sources_from_results(results),
        "guard": {"reason": "ok"},
    }

# ============================================================
# /ask
# ============================================================
@app.post("/ask")
def ask(req: ChatRequest):
    intent = classify_intent(req.question, llm)

    if intent.intent == "command":
        return command(req)

    return chat(req)
