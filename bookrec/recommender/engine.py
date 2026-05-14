from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .catalog import load_books_themes, compute_catalog_mass
from .embeddings import ThemeEmbeddings
from .recommend import recommend

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

log = logging.getLogger(__name__)

RECSYS_VERSION = "themes_v1"


class RecommendEngine:

    def __init__(self) -> None:
        self._books_themes: dict[int, dict[str, float]] = {}
        self._catalog_mass: dict[str, float] = {}
        self._books_meta: dict[int, dict] = {}
        self._embeddings = ThemeEmbeddings()
        self._ready = False

    def load(self, conn: "PGConnection") -> None:

        self._books_themes = load_books_themes(conn)
        self._catalog_mass = compute_catalog_mass(self._books_themes)
        self._books_meta = _load_books_meta(conn)
        self._embeddings.build()
        self._ready = True
        log.info(
            "RecommendEngine ready: %d books with themes, %d in meta cache",
            len(self._books_themes),
            len(self._books_meta),
        )

    def refresh(self, conn: "PGConnection") -> None:
        self._books_themes = load_books_themes(conn)
        self._catalog_mass = compute_catalog_mass(self._books_themes)
        self._books_meta = _load_books_meta(conn)
        log.info("Catalog refreshed: %d books with themes", len(self._books_themes))

    def recommend_for_user(
        self,
        conn: "PGConnection",
        db_user_id: int,
        top_k: int = 20,
        already_seen: set[int] | None = None,
    ) -> list[dict]:
      
        if not self._ready:
            log.warning("Engine not loaded ")
            return []

        liked_ids = _get_liked_book_ids(conn, db_user_id)

        if not liked_ids:
            return self._popularity_fallback(top_k, already_seen)

        ranked = recommend(
            user_book_ids=liked_ids,
            books_themes=self._books_themes,
            catalog_mass=self._catalog_mass,
            theme_matrix=self._embeddings.matrix,
            theme_index=self._embeddings.index,
            excluded_ids=already_seen,
            top_k=top_k,
        )

        return self._enrich(ranked)

    def popularity_fallback(self, top_k: int = 20, excluded: set[int] | None = None) -> list[dict]:
        return self._popularity_fallback(top_k, excluded)


    def _enrich(self, ranked: list[tuple[int, float]]) -> list[dict]:
        """Attach book metadata"""
        results = []
        for book_id, score in ranked:
            meta = self._books_meta.get(book_id)
            if meta is None:
                continue
            results.append({
                "item_id": book_id,
                "title": meta["title"],
                "author": meta.get("author", ""),
                "description": meta.get("description") or "",
                "genre": meta.get("genre", ""),
                "score": round(score, 4),
            })
        return results

    def _popularity_fallback(
        self,
        top_k: int,
        excluded: set[int] | None = None,
    ) -> list[dict]:
        """Return highest-rated books the user hasn't seen yet."""
        excluded = excluded or set()
        ranked = [
            (bid, meta["rating"] or 0.0)
            for bid, meta in self._books_meta.items()
            if bid not in excluded
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return self._enrich(ranked[:top_k])


def _load_books_meta(conn: "PGConnection") -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, title, author, genre, description, rating
            FROM books
            WHERE title IS NOT NULL
        """)
        rows = cur.fetchall()
    return {
        r[0]: {
            "title":       r[1],
            "author":      r[2] or "",
            "genre":       r[3] or "",
            "description": r[4] or "",
            "rating":      float(r[5]) if r[5] else None,
        }
        for r in rows
    }


def _get_liked_book_ids(conn: "PGConnection", user_id: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT book_id FROM saved_books WHERE user_id = %s
            UNION
            SELECT book_id FROM user_interactions
            WHERE user_id = %s AND event IN ('like', 'save')
        """, (user_id, user_id))
        return [r[0] for r in cur.fetchall()]
