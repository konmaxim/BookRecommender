from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..config import cfg
from ..db import (
    upsert_user_idea_profile,
    load_user_idea_profile,
    get_user_idea_profile_freshness,
)
from .catalog import load_books_themes, compute_catalog_mass
from .embeddings import ThemeEmbeddings
from .idea_catalog import (
    load_book_ideas,
    load_canonical_centroids,
)
from .idea_profile import build_user_idea_profile
from .recommend import recommend

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

log = logging.getLogger(__name__)

RECSYS_VERSION = "themes_v1_ideas_v1"


class RecommendEngine:

    def __init__(self) -> None:
        self._books_themes: dict[int, dict[str, float]] = {}
        self._catalog_mass: dict[str, float] = {}
        self._books_meta: dict[int, dict] = {}
        self._embeddings = ThemeEmbeddings()

        # idea channel state — empty when data files are missing
        self._book_ideas: dict[int, list[dict]] = {}
        self._centroids: np.ndarray | None = None
        self._idea_channel_ready = False

        self._ready = False

    def load(self, conn: "PGConnection") -> None:
        self._books_themes = load_books_themes(conn)
        self._catalog_mass = compute_catalog_mass(self._books_themes)
        self._books_meta = _load_books_meta(conn)
        self._embeddings.build()
        self._load_idea_channel(conn)
        self._ready = True
        log.info(
            "RecommendEngine ready: %d books with themes, %d in meta cache, idea_channel=%s",
            len(self._books_themes),
            len(self._books_meta),
            self._idea_channel_ready,
        )

    def refresh(self, conn: "PGConnection") -> None:
        self._books_themes = load_books_themes(conn)
        self._catalog_mass = compute_catalog_mass(self._books_themes)
        self._books_meta = _load_books_meta(conn)
        self._load_idea_channel(conn)
        log.info("Catalog refreshed: %d books with themes", len(self._books_themes))

    def _load_idea_channel(self, conn: "PGConnection") -> None:
        try:
            self._centroids = load_canonical_centroids()
            self._book_ideas = load_book_ideas()
        except FileNotFoundError as e:
            log.warning("Idea channel disabled: %s", e)
            self._centroids = None
            self._book_ideas = {}
            self._idea_channel_ready = False
            return
        self._idea_channel_ready = True

    def refresh_user_idea_profile(
        self,
        conn: "PGConnection",
        user_id: int,
    ) -> list[dict]:
        """Rebuild from the saved library and persist. Returns the new profile."""
        if not self._idea_channel_ready:
            return []
        liked_ids = _get_liked_book_ids(conn, user_id)
        profile = build_user_idea_profile(liked_ids, self._book_ideas)
        upsert_user_idea_profile(conn, user_id, profile)
        return profile

    def _ensure_user_idea_profile(
        self,
        conn: "PGConnection",
        user_id: int,
    ) -> list[dict]:
        """
        Lazy invalidation: if the persisted profile is missing or older
        than the user's latest interaction, rebuild.
        """
        if not self._idea_channel_ready:
            return []
        profile_ts, library_ts = get_user_idea_profile_freshness(conn, user_id)
        if profile_ts is None or (library_ts is not None and library_ts > profile_ts):
            return self.refresh_user_idea_profile(conn, user_id)
        return load_user_idea_profile(conn, user_id)

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

        idea_profile = self._ensure_user_idea_profile(conn, db_user_id) if self._idea_channel_ready else []

        ranked = recommend(
            user_book_ids=liked_ids,
            books_themes=self._books_themes,
            catalog_mass=self._catalog_mass,
            theme_matrix=self._embeddings.matrix,
            theme_index=self._embeddings.index,
            excluded_ids=already_seen,
            top_k=top_k,
            idea_profile=idea_profile if self._idea_channel_ready else None,
            book_ideas=self._book_ideas if self._idea_channel_ready else None,
            centroids=self._centroids,
            gamma=cfg.idea_gamma,
            lam=cfg.idea_lambda,
            beta=cfg.idea_beta,
            sim_floor=cfg.idea_sim_floor,
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
