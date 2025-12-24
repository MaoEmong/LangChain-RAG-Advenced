# docstore_sqlite.py
# ============================================================
# SQLite 기반 DocStore (ParentDocumentRetriever용 - BaseStore 호환)
# - BaseStore 요구 메서드: mget, mset, mdelete, yield_keys
# ============================================================

import sqlite3
import json
from typing import Iterable, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class SQLiteDocStore(BaseStore[str, Document]):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
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
        payload = {
            "page_content": doc.page_content,
            "metadata": doc.metadata or {},
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _de(s: str) -> Document:
        payload = json.loads(s)
        return Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {}) or {},
        )

    # ------------------------------------------------------------
    # BaseStore required methods
    # ------------------------------------------------------------
    def mset(self, key_value_pairs: Iterable[Tuple[str, Document]]) -> None:
        pairs = list(key_value_pairs)
        if not pairs:
            return

        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO docs (k, v) VALUES (?, ?)",
                [(k, self._ser(v)) for k, v in pairs],
            )

    def mget(self, keys: Iterable[str]) -> List[Optional[Document]]:
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
