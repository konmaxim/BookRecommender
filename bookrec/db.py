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
