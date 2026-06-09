from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .db import (
    get_db, create_tables, upsert_user, get_user, get_top_books_by_genre,
    save_books, log_interaction, log_recommendation_event, get_user_library,
)
from .recommender import engine
from .recommender.engine import RECSYS_VERSION

log = logging.getLogger(__name__)

#TODO нормальную регистрацию и авторизацию с валидацией, паролями и тд, сейчас фокусируюсь над ML-частью

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading recommendation engine...")
    try:
        with get_db() as conn:
            create_tables(conn)
            engine.load(conn)
    except Exception:
        log.exception("Engine failed to load — recommendations will be unavailable")
    yield
    log.info("Shutting down.")


app = FastAPI(title="BookRec API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
#validate
_NAME_RE = re.compile(r"^[\w\s'\-]{2,50}$", re.UNICODE)


def _validate_name(name: str) -> str | None:
    if not name or not name.strip():
        return "Введите имя."
    name = name.strip()
    if not _NAME_RE.match(name):
        return "Неправильный формат."
    return None

class NameBody(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        err = _validate_name(v)
        if err:
            raise ValueError(err)
        return v.strip()


class SaveBooksBody(BaseModel):
    user_id: int
    book_ids: list[int]


class InteractBody(BaseModel):
    user_id: str
    db_user_id: Optional[int] = None
    item_id: int
    event: str                          # 'like' | 'skip' | 'save' | 'shown'
    session_id: Optional[str] = None
    rank_position: Optional[int] = None
    recsys_version: Optional[str] = None


@app.post("/register", status_code=201)
def register(body: NameBody):
    try:
        with get_db() as conn:
            create_tables(conn)
            existing = get_user(conn, body.name)
            if existing:
                raise HTTPException(status_code=409, detail="Это имя уже занято, выберите другое. ")
            user_id = upsert_user(conn, body.name)
    except HTTPException:
        raise
    except Exception:
        log.exception("register failed")
        raise HTTPException(status_code=500, detail="Server error.")

    log.info("Registered user %r (id=%s)", body.name, user_id)
    return {"id": user_id, "name": body.name}


@app.post("/login")
def login(body: NameBody):
    try:
        with get_db() as conn:
            user = get_user(conn, body.name)
            if user is None:
                raise HTTPException(status_code=404, detail="Ваш аккаунт не найден в базе.")
    except HTTPException:
        raise
    except Exception:
        log.exception("login failed")
        raise HTTPException(status_code=500, detail="Server error.")

    log.info("Logged in user %r (id=%s)", user.name, user.id)
    return {"id": user.id, "name": user.name}


@app.get("/books")
def books(per_genre: Optional[int] = Query(default=None, ge=1)):
    try:
        with get_db() as conn:
            create_tables(conn)
            data = get_top_books_by_genre(conn, per_genre=per_genre)
        return data
    except Exception:
        log.exception("GET /books failed")
        raise HTTPException(status_code=500, detail="Server error.")


@app.post("/save-books")
def save_books_route(body: SaveBooksBody):
    if not body.book_ids:
        raise HTTPException(status_code=400, detail="book_ids must be non-empty")
    try:
        with get_db() as conn:
            n = save_books(conn, body.user_id, body.book_ids)
            try:
                engine.refresh_user_idea_profile(conn, body.user_id)
            except Exception:
                log.exception("refresh_user_idea_profile failed (will rebuild lazily next request)")
        return {"saved": n}
    except Exception:
        log.exception("save-books failed")
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/recommendations")
def recommendations(
    db_user_id: Optional[int] = Query(default=None),
    top_k: int = Query(default=20, ge=1, le=100),
    seen: str = Query(default=""),   
):
    already_seen: set[int] = set()
    if seen:
        try:
            already_seen = {int(x) for x in seen.split(",") if x.strip()}
        except ValueError:
            pass

    try:
        if db_user_id is not None:
            with get_db() as conn:
                results = engine.recommend_for_user(
                    conn, db_user_id, top_k=top_k, already_seen=already_seen
                )
        else:
            results = engine.popularity_fallback(top_k=top_k, excluded=already_seen)

        session_id = str(uuid.uuid4())
        for rank, book_dict in enumerate(results):
            book_dict["rank_position"] = rank
            book_dict["session_id"] = session_id
            book_dict["recsys_version"] = RECSYS_VERSION
        return {"session_id": session_id, "version": RECSYS_VERSION, "items": results}
    except Exception:
        log.exception("recommendations failed")
        raise HTTPException(status_code=500, detail="Server error")


_EVENT_MAP = {"like": "liked", "skip": "skipped", "save": "saved", "shown": "shown"}

@app.post("/interact")
def interact(body: InteractBody):
    if body.event not in _EVENT_MAP:
        raise HTTPException(status_code=400, detail="event must be like, skip, save, or shown")
    try:
        with get_db() as conn:
            # user_interactions only tracks like/skip/save (not shown)
            if body.event in ("like", "skip", "save"):
                log_interaction(
                    conn,
                    book_id=body.item_id,
                    event=body.event,
                    user_id=body.db_user_id,
                    anon_id=body.user_id,
                )
            # recommendation_events tracks all event types for logged-in users
            if body.db_user_id is not None:
                log_recommendation_event(
                    conn,
                    user_id=body.db_user_id,
                    book_id=body.item_id,
                    event_type=_EVENT_MAP[body.event],
                    recsys_version=body.recsys_version or RECSYS_VERSION,
                    rank_position=body.rank_position,
                    session_id=body.session_id,
                )
            # liked/saved change the library → eagerly rebuild the idea profile
            if body.db_user_id is not None and body.event in ("like", "save"):
                try:
                    engine.refresh_user_idea_profile(conn, body.db_user_id)
                except Exception:
                    log.exception("refresh_user_idea_profile failed (will rebuild lazily next request)")
    except Exception:
        log.exception("interact failed")
    return {"ok": True}


@app.get("/library")
def library(db_user_id: int = Query(...)):
    try:
        with get_db() as conn:
            return get_user_library(conn, db_user_id)
    except Exception:
        log.exception("library failed")
        raise HTTPException(status_code=500, detail="Server error")


@app.post("/refresh-engine")
def refresh_engine():
    """Reload catalog data into the engine (call after adding new books/themes)."""
    try:
        with get_db() as conn:
            engine.refresh(conn)
        return {"ok": True, "catalog_size": len(engine._books_themes)}
    except Exception:
        log.exception("refresh-engine failed")
        raise HTTPException(status_code=500, detail="Server error")

def run() -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    uvicorn.run("bookrec.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
