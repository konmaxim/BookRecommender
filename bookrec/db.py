from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import psycopg2
import psycopg2.extras

from .config import cfg
from .models import Book, User, UserRating

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS books (
    id              SERIAL PRIMARY KEY,
    book_url        TEXT         NOT NULL UNIQUE,
    title           TEXT         NOT NULL,
    author          TEXT,
    genre           TEXT,
    rating          NUMERIC(3,1),
    isbn            TEXT,
    year            INTEGER,
    publisher       TEXT,
    language        TEXT,
    readers_count   INTEGER,
    reviews_count   INTEGER,
    quotes_count    INTEGER,
    description     TEXT,
    tags            JSONB,
    scraped_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_books_author ON books (author);
CREATE INDEX IF NOT EXISTS idx_books_rating ON books (rating DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_books_year   ON books (year);
CREATE INDEX IF NOT EXISTS idx_books_genre  ON books (genre);
CREATE INDEX IF NOT EXISTS idx_books_tags   ON books USING gin (tags);

CREATE TABLE IF NOT EXISTS saved_books (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    book_id    INTEGER NOT NULL,
    saved_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, book_id)
);
CREATE INDEX IF NOT EXISTS idx_saved_books_user ON saved_books (user_id);

CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_ratings (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    book_id     INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    rating      SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    rated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, book_id)
);
CREATE INDEX IF NOT EXISTS idx_user_ratings_user ON user_ratings (user_id);
CREATE INDEX IF NOT EXISTS idx_user_ratings_book ON user_ratings (book_id);

CREATE TABLE IF NOT EXISTS reviews (
    id          SERIAL PRIMARY KEY,
    book_id     INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    reviewer    TEXT,
    rating      SMALLINT,
    review_date TEXT,
    likes_count INTEGER,
    review_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_interactions (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    anon_id     TEXT,
    book_id     INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    event       TEXT NOT NULL CHECK (event IN ('like', 'skip', 'save')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_interactions_user   ON user_interactions (user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_anon   ON user_interactions (anon_id);

CREATE TABLE IF NOT EXISTS recommendation_events (
    event_id        BIGSERIAL PRIMARY KEY,
    user_id         INT NOT NULL,
    book_id         INT NOT NULL,
    event_type      TEXT NOT NULL,  -- 'shown', 'liked', 'skipped', 'saved'
    rank_position   INT,
    recsys_version  TEXT NOT NULL,
    session_id      TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rec_events_user    ON recommendation_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_rec_events_version ON recommendation_events (recsys_version, event_type);
CREATE INDEX IF NOT EXISTS idx_rec_events_session ON recommendation_events (session_id);

CREATE TABLE IF NOT EXISTS user_idea_profile (
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    canonical_idea_id INTEGER NOT NULL,
    engagement        DOUBLE PRECISION NOT NULL,
    net_stance        DOUBLE PRECISION NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, canonical_idea_id)
);
CREATE INDEX IF NOT EXISTS idx_uip_user ON user_idea_profile (user_id);
"""


_UPSERT_BOOK = """
INSERT INTO books (
    book_url, title, author, genre, rating, isbn, year, publisher, language,
    readers_count, reviews_count, quotes_count, description
) VALUES (
    %(book_url)s, %(title)s, %(author)s, %(genre)s, %(rating)s, %(isbn)s,
    %(year)s, %(publisher)s, %(language)s, %(readers_count)s,
    %(reviews_count)s, %(quotes_count)s, %(description)s
)
ON CONFLICT (book_url) DO UPDATE SET
    title           = EXCLUDED.title,
    author          = EXCLUDED.author,
    genre           = EXCLUDED.genre,
    rating          = EXCLUDED.rating,
    isbn            = EXCLUDED.isbn,
    year            = EXCLUDED.year,
    publisher       = EXCLUDED.publisher,
    language        = EXCLUDED.language,
    readers_count   = EXCLUDED.readers_count,
    reviews_count   = EXCLUDED.reviews_count,
    quotes_count    = EXCLUDED.quotes_count,
    description     = EXCLUDED.description,
    updated_at      = NOW()
RETURNING (xmax = 0) AS inserted;
"""

_INSERT_REVIEW = """
INSERT INTO reviews (book_id, reviewer, rating, review_date, likes_count, review_text)
VALUES (%s, %s, %s, %s, %s, %s)
"""

_UPSERT_USER = """
INSERT INTO users (name) VALUES (%s)
ON CONFLICT (name) DO NOTHING
RETURNING id;
"""

_GET_USER_ID = "SELECT id FROM users WHERE name = %s;"

_UPSERT_USER_RATING = """
INSERT INTO user_ratings (user_id, book_id, rating)
VALUES (%s, %s, %s)
ON CONFLICT (user_id, book_id) DO UPDATE SET
    rating   = EXCLUDED.rating,
    rated_at = NOW();
"""

_GET_USER_RATINGS = """
SELECT ur.book_id, ur.rating, b.title, b.book_url
FROM user_ratings ur
JOIN books b ON b.id = ur.book_id
WHERE ur.user_id = %s
ORDER BY ur.rated_at DESC;
"""



def get_connection() -> "PGConnection":
    conn = psycopg2.connect(**cfg.db.as_dict())
    conn.autocommit = False
    log.info("Connected to %s@%s/%s", cfg.db.user, cfg.db.host, cfg.db.dbname)
    return conn


@contextmanager
def get_db() -> Generator["PGConnection", None, None]:
    """Context manager that yields a connection and always closes it."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()



def create_tables(conn: "PGConnection") -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    log.info("Tables ready")



def upsert_books(conn: "PGConnection", books: list[Book]) -> tuple[int, int]:
    inserted = updated = 0
    with conn.cursor() as cur:
        for book in books:
            record = {
                "book_url":      book.book_url,
                "title":         book.title,
                "author":        book.author,
                "genre":         book.genre,
                "rating":        book.rating,
                "isbn":          book.isbn,
                "year":          book.year,
                "publisher":     book.publisher,
                "language":      book.language,
                "readers_count": book.readers_count,
                "reviews_count": book.reviews_count,
                "quotes_count":  book.quotes_count,
                "description":   book.description,
            }
            try:
                cur.execute(_UPSERT_BOOK, record)
                row = cur.fetchone()
                if row and row[0]:
                    inserted += 1
                else:
                    updated += 1
            except psycopg2.Error as exc:
                log.error("Upsert failed for '%s': %s", book.title, exc)
                conn.rollback()
    conn.commit()
    return inserted, updated


def insert_reviews(
    conn: "PGConnection",
    book_db_id: int,
    reviews: list[tuple],
) -> int:
    with conn.cursor() as cur:
        for (reviewer, rating, review_date, likes_count, text) in reviews:
            cur.execute(_INSERT_REVIEW, (book_db_id, reviewer, rating, review_date, likes_count, text))
    conn.commit()
    return len(reviews)





def upsert_user(conn: "PGConnection", name: str) -> int:
    """Insert user by name (case-sensitive); return their DB id."""
    with conn.cursor() as cur:
        cur.execute(_UPSERT_USER, (name,))
        row = cur.fetchone()
        if row is None:
            cur.execute(_GET_USER_ID, (name,))
            row = cur.fetchone()
    conn.commit()
    return row[0]


#rating options for later, TODO 

def upsert_user_rating(
    conn: "PGConnection",
    user_id: int,
    book_id: int,
    rating: int,
) -> None:
    """Add or update a user's 1-5 rating for a book."""
    with conn.cursor() as cur:
        cur.execute(_UPSERT_USER_RATING, (user_id, book_id, rating))
    conn.commit()


def get_user_ratings(conn: "PGConnection", user_id: int) -> list[UserRating]:
    """Return all books rated by a user, most recent first."""
    with conn.cursor() as cur:
        cur.execute(_GET_USER_RATINGS, (user_id,))
        rows = cur.fetchall()
    return [
        UserRating(book_id=r[0], rating=r[1], book_title=r[2], book_url=r[3])
        for r in rows
    ]


def save_books(conn: "PGConnection", user_id: int, book_ids: list[int]) -> int:
    """Insert book_ids into saved_books for user; ignore duplicates. Returns count saved."""
    if not book_ids:
        return 0
    with conn.cursor() as cur:
        for book_id in book_ids:
            cur.execute(
                "INSERT INTO saved_books (user_id, book_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, book_id),
            )
    conn.commit()
    return len(book_ids)


def get_saved_book_ids(conn: "PGConnection", user_id: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT book_id FROM saved_books WHERE user_id = %s ORDER BY saved_at DESC", (user_id,))
        return [r[0] for r in cur.fetchall()]


def get_top_books_by_genre(conn: "PGConnection", per_genre: int | None = None) -> list[dict]:
    """Return books grouped by genre, ordered by rating.

    If per_genre is None (default), all books are returned.
    Otherwise returns up to per_genre books per genre.
    """
    with conn.cursor() as cur:
        if per_genre is None:
            cur.execute("""
                SELECT id, title, author, genre, rating
                FROM books
                WHERE genre IS NOT NULL AND title IS NOT NULL
                ORDER BY genre, rating DESC NULLS LAST, readers_count DESC NULLS LAST
            """)
        else:
            cur.execute("""
                SELECT id, title, author, genre, rating
                FROM (
                    SELECT id, title, author, genre, rating,
                           ROW_NUMBER() OVER (
                               PARTITION BY genre
                               ORDER BY rating DESC NULLS LAST, readers_count DESC NULLS LAST
                           ) AS rn
                    FROM books
                    WHERE genre IS NOT NULL AND title IS NOT NULL
                ) ranked
                WHERE rn <= %s
                ORDER BY genre, rn
            """, (per_genre,))
        rows = cur.fetchall()
    return [
        {"book_id": r[0], "title": r[1], "author": r[2] or "", "genre": r[3], "rating": float(r[4]) if r[4] else None}
        for r in rows
    ]


def log_interaction(
    conn: "PGConnection",
    book_id: int,
    event: str,
    user_id: int | None = None,
    anon_id: str | None = None,
) -> None:
    """Record a like/skip/save interaction from the feed."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_interactions (user_id, anon_id, book_id, event)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, anon_id, book_id, event),
        )
    conn.commit()


def log_recommendation_event(
    conn: "PGConnection",
    user_id: int,
    book_id: int,
    event_type: str,
    recsys_version: str,
    rank_position: int | None = None,
    session_id: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recommendation_events
                (user_id, book_id, event_type, rank_position, recsys_version, session_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, book_id, event_type, rank_position, recsys_version, session_id),
        )
    conn.commit()


def get_user_library(conn: "PGConnection", user_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT b.id, b.title, b.author, b.genre
            FROM saved_books sb JOIN books b ON b.id = sb.book_id
            WHERE sb.user_id = %s ORDER BY sb.saved_at DESC
            """,
            (user_id,),
        )
        chosen = [
            {"book_id": r[0], "title": r[1], "author": r[2] or "", "genre": r[3]}
            for r in cur.fetchall()
        ]

        def _fetch_event(event_type):
            cur.execute(
                """
                SELECT DISTINCT ON (re.book_id) b.id, b.title, b.author, b.genre
                FROM recommendation_events re JOIN books b ON b.id = re.book_id
                WHERE re.user_id = %s AND re.event_type = %s
                ORDER BY re.book_id, re.created_at DESC
                """,
                (user_id, event_type),
            )
            return [
                {"book_id": r[0], "title": r[1], "author": r[2] or "", "genre": r[3]}
                for r in cur.fetchall()
            ]

        liked = _fetch_event("liked")
        saved = _fetch_event("saved")
    return {"chosen": chosen, "liked": liked, "saved": saved}


def get_user(conn: "PGConnection", name: str) -> User | None:
    """Load a User object (with ratings) by name, or None if not found."""
    with conn.cursor() as cur:
        cur.execute(_GET_USER_ID, (name,))
        row = cur.fetchone()
    if row is None:
        return None
    user_id = row[0]
    ratings = get_user_ratings(conn, user_id)
    return User(name=name, id=user_id, ratings=ratings)


def upsert_user_idea_profile(
    conn: "PGConnection",
    user_id: int,
    items: list[dict],
) -> None:
    """
    Replace this user's profile rows in one shot. `items` is a list of
    {cid, engagement, net_stance} dicts (matching build_user_idea_profile output).
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_idea_profile WHERE user_id = %s", (user_id,))
        if items:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO user_idea_profile
                    (user_id, canonical_idea_id, engagement, net_stance)
                VALUES %s
                """,
                [
                    (user_id, int(it["cid"]), float(it["a"]), float(it["p"]))
                    for it in items
                ],
            )
    conn.commit()


def load_user_idea_profile(
    conn: "PGConnection",
    user_id: int,
) -> list[dict]:
    """Returns [{cid, a (engagement), p (net_stance)}, ...]."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT canonical_idea_id, engagement, net_stance
            FROM user_idea_profile
            WHERE user_id = %s
            """,
            (user_id,),
        )
        return [
            {"cid": int(cid), "a": float(a), "p": float(p)}
            for cid, a, p in cur.fetchall()
        ]


def get_user_idea_profile_freshness(
    conn: "PGConnection",
    user_id: int,
) -> tuple[object | None, object | None]:
    """
    Returns (max(updated_at) on profile rows, max(created_at) on
    user_interactions). Caller checks: if profile is None or older
    than the latest interaction, rebuild.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(updated_at) FROM user_idea_profile WHERE user_id = %s",
            (user_id,),
        )
        profile_ts = cur.fetchone()[0]
        cur.execute(
            """
            SELECT GREATEST(
                COALESCE((SELECT MAX(created_at) FROM user_interactions WHERE user_id = %s), 'epoch'),
                COALESCE((SELECT MAX(saved_at)   FROM saved_books       WHERE user_id = %s), 'epoch')
            )
            """,
            (user_id, user_id),
        )
        library_ts = cur.fetchone()[0]
    return profile_ts, library_ts
