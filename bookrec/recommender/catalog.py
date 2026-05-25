from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection


def load_books_themes(
    conn: "PGConnection",
    length_norm_alpha: float = 0.75,
) -> dict[int, dict[str, float]]:
    """
    Pivoted length normalization for longer descriptions to be retrieved less :

        normalized[t] = raw[t] / ((1 - alpha) * pivot + alpha * length)

    where length is the desc's L2 and pivot is the catalog mean.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, tags->'topics' AS topics
            FROM books
            WHERE tags IS NOT NULL
              AND tags ? 'topics'
              AND jsonb_array_length(tags->'topics') > 0
        """)
        rows = cur.fetchall()

    raw: dict[int, dict[str, float]] = {}
    lengths: dict[int, float] = {}
    for book_id, topics in rows:
        if not topics:
            continue
        themes = {
            t["label"]: float(t["score"])
            for t in topics
            if isinstance(t, dict) and "label" in t and "score" in t
        }
        if not themes:
            continue
        raw[book_id] = themes
        lengths[book_id] = sum(v ** 2 for v in themes.values()) ** 0.5

    if not raw:
        return {}

    pivot = sum(lengths.values()) / len(lengths)

    books_themes: dict[int, dict[str, float]] = {}
    for book_id, themes in raw.items():
        divisor = (1 - length_norm_alpha) * pivot + length_norm_alpha * lengths[book_id]
        if divisor <= 0:
            books_themes[book_id] = themes
        else:
            books_themes[book_id] = {k: v / divisor for k, v in themes.items()}

    return books_themes


def compute_catalog_mass(
    #how prevelant is a theme across the entire catalog, will be used 
    #to damp out themes that are common across many books
    books_themes: dict[int, dict[str, float]],
) -> dict[str, float]:
    theme_totals: dict[str, float] = defaultdict(float)
    grand_total = 0.0
    for themes in books_themes.values():
        for theme, score in themes.items():
            theme_totals[theme] += score
            grand_total += score
    if grand_total == 0:
        return {}
    return {t: total / grand_total for t, total in theme_totals.items()}
