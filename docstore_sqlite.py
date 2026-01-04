"""
docstore_sqlite.py
============================================================
SQLite 기반 DocStore (ParentDocumentRetriever용)

LangChain의 BaseStore 인터페이스를 구현한 SQLite 기반 문서 저장소입니다.
Parent-Child 문서 구조에서 Parent 문서를 저장하는 데 사용됩니다.

특징:
- BaseStore 호환: LangChain의 BaseStore 인터페이스 구현
- Parent 문서 저장: 파일 전체 텍스트를 저장
- SQLite 백엔드: 가볍고 안정적인 로컬 저장소

필수 메서드:
- mget: 여러 문서 조회
- mset: 여러 문서 저장
- mdelete: 여러 문서 삭제
- yield_keys: 모든 키 순회
"""

import sqlite3
import json
from typing import Iterable, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class SQLiteDocStore(BaseStore[str, Document]):
    """
    SQLite 기반 문서 저장소
    
    LangChain의 BaseStore 인터페이스를 구현하여
    Parent 문서를 SQLite 데이터베이스에 저장합니다.
    """
    
    def __init__(self, db_path: str):
        """
        SQLiteDocStore 초기화
        
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._init()

    def _conn(self):
        """
        SQLite 데이터베이스 연결을 생성합니다.
        
        Returns:
            sqlite3.Connection: 데이터베이스 연결 객체
        """
        return sqlite3.connect(self.db_path)

    def _init(self):
        """
        데이터베이스 테이블을 초기화합니다.
        
        docs 테이블이 없으면 생성합니다.
        - k: 문서 ID (PRIMARY KEY)
        - v: 문서 내용 (JSON 문자열)
        """
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _ser(doc: Document) -> str:
        """
        Document 객체를 JSON 문자열로 직렬화합니다.
        
        Args:
            doc: 직렬화할 Document 객체
        
        Returns:
            str: JSON 문자열
        """
        payload = {
            "page_content": doc.page_content,
            "metadata": doc.metadata or {},
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _de(s: str) -> Document:
        """
        JSON 문자열을 Document 객체로 역직렬화합니다.
        
        Args:
            s: JSON 문자열
        
        Returns:
            Document: Document 객체
        """
        payload = json.loads(s)
        return Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {}) or {},
        )

    # ------------------------------------------------------------
    # BaseStore 필수 메서드
    # ------------------------------------------------------------
    def mset(self, key_value_pairs: Iterable[Tuple[str, Document]]) -> None:
        """
        여러 문서를 저장합니다 (BaseStore 필수 메서드).
        
        Args:
            key_value_pairs: (키, Document) 튜플의 반복 가능 객체
                - 키: 문서 ID (doc_id)
                - Document: 저장할 문서 객체
        
        Note:
            - 이미 존재하는 키의 경우 기존 문서를 덮어씁니다 (INSERT OR REPLACE)
            - 빈 리스트인 경우 아무 작업도 하지 않습니다
        """
        pairs = list(key_value_pairs)
        if not pairs:
            return

        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO docs (k, v) VALUES (?, ?)",
                [(k, self._ser(v)) for k, v in pairs],
            )

    def mget(self, keys: Iterable[str]) -> List[Optional[Document]]:
        """
        여러 문서를 조회합니다 (BaseStore 필수 메서드).
        
        Args:
            keys: 조회할 문서 ID 리스트
        
        Returns:
            List[Optional[Document]]: Document 객체 리스트
                - 키가 존재하면 Document 객체
                - 키가 없으면 None
                - 입력 키 순서와 동일한 순서로 반환
        """
        keys = list(keys)
        if not keys:
            return []

        with self._conn() as con:
            cur = con.execute(
                f"SELECT k, v FROM docs WHERE k IN ({','.join(['?'] * len(keys))})",
                keys,
            )
            rows = {k: v for k, v in cur.fetchall()}

        out: List[Optional[Document]] = []
        for k in keys:
            s = rows.get(k)
            out.append(self._de(s) if s is not None else None)
        return out

    def mdelete(self, keys: Iterable[str]) -> None:
        """
        여러 문서를 삭제합니다 (BaseStore 필수 메서드).
        
        Args:
            keys: 삭제할 문서 ID 리스트
        
        Note:
            - 존재하지 않는 키는 무시됩니다
            - 빈 리스트인 경우 아무 작업도 하지 않습니다
        """
        keys = list(keys)
        if not keys:
            return

        with self._conn() as con:
            con.execute(
                f"DELETE FROM docs WHERE k IN ({','.join(['?'] * len(keys))})",
                keys,
            )

    def yield_keys(self) -> Iterator[str]:
        """
        BaseStore가 요구하는 추상 메서드.
        저장된 모든 key를 순회하는 iterator를 반환한다.
        """
        with self._conn() as con:
            cur = con.execute("SELECT k FROM docs")
            rows = cur.fetchall()
        for (k,) in rows:
            yield k
