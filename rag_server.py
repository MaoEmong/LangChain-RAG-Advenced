from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from chains.rag_chain import build_rag_chain
from chains.command_chain import build_command_chain

from prompts.command_prompt import COMMAND_PROMPT_TEMPLATE

from services.confidence import calculate_confidence
from services.vector_store import create_vector_store
from services.command_parser import parse_command_json
from services.confidence import calculate_confidence
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

app = FastAPI()

# Vector DB / Retriever
vector_db = create_vector_store(
    OPENAI_API_KEY,
    EMBED_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": max(12, TOP_K * 3),
    },
)

# LLM
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)

# Prompt
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

command_prompt = ChatPromptTemplate.from_template(COMMAND_PROMPT_TEMPLATE)
command_chain = build_command_chain(retriever, llm, command_prompt)

rag_chain = build_rag_chain(retriever, llm, prompt)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    results = vector_db.similarity_search_with_score(req.question, k=TOP_K)

    # 1️⃣ 검색 결과 없음
    if not results:
        return {
            "type": "rag_answer",
            "question": req.question,
            "answer": "문서에서 근거를 찾지 못했습니다.",
            "sources": [],
            "guard": {"reason": "no_results"},
        }

    top_score = float(results[0][1])
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)

    # 2️⃣ TOP1 점수 컷
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

    # 3️⃣ 충분히 좋은 문서 수 부족
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

    # ============================
    # ✅ 여기부터가 "정상 성공 경로"
    # ============================

    # 답변 생성
    answer = rag_chain.invoke(req.question)
    confidence = calculate_confidence(top_score, good_hits)
    # source 정보 구성
    sources = []
    for d, score in results:
        sources.append({
            "source": d.metadata.get("source"),
            "score": float(score),
            "preview": d.page_content[:200],
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
    # 1) 기존 RAG 검색 + confidence 계산
    results = vector_db.similarity_search_with_score(req.question, k=TOP_K)

    if not results:
        return {
            "type": "command",
            "speech": "실행 가능한 명령을 찾지 못했습니다.",
            "actions": [],
            "confidence": {"level": "low"},
        }

    top_score = float(results[0][1])
    good_hits = sum(1 for _, s in results if float(s) <= GOOD_HIT_SCORE_MAX)
    confidence = calculate_confidence(top_score, good_hits)
    COMMAND_HIGH_THRESHOLD = 0.65
    # 2) confidence가 낮으면 바로 차단
    if confidence["score"] < COMMAND_HIGH_THRESHOLD:
        return {
            "type": "command",
            "speech": "확신이 부족하여 명령을 실행할 수 없습니다.",
            "actions": [],
            "confidence": confidence,
        }

    # 3) Command JSON 생성
    raw_text = command_chain.invoke(req.question)

    # 4) JSON 파싱/검증
    parsed = parse_command_json(raw_text)

    if not parsed:
        return {
            "type": "command",
            "speech": "명령을 해석하지 못했습니다.",
            "actions": [],
            "confidence": confidence,
        }

    # 🔒 화이트리스트 검증
    ok, reason = validate_commands(parsed)

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

    # 5) 최종 안전한 command 반환
    return {
        "type": "command",
        "speech": parsed.speech,
        "actions": [a.model_dump() for a in parsed.actions],
        "confidence": confidence,
    }

@app.post("/ask")
def ask(req: ChatRequest):
    intent = classify_intent(req.question, llm)

    if intent.intent == "command":
        # 기존 /command 로직을 함수로 빼두었다면 그걸 호출하는 게 베스트
        # 일단은 command 엔드포인트 로직을 그대로 여기로 옮겨도 됨
        return command(req)  # 이미 만들어둔 /command 함수 재사용 가능

    # explain이면 기존 /chat 로직
    return chat(req)