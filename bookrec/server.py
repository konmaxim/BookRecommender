from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .db import get_db, create_tables, upsert_user, get_user, get_top_books_by_genre, save_books, log_interaction
from .recommender import engine

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
    event: str


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
        return results
    except Exception:
        log.exception("recommendations failed")
        raise HTTPException(status_code=500, detail="Server error")


@app.post("/interact")
def interact(body: InteractBody):
    if body.event not in ("like", "skip", "save"):
        raise HTTPException(status_code=400, detail="event must be like, skip, or save")
    try:
        with get_db() as conn:
            log_interaction(
                conn,
                book_id=body.item_id,
                event=body.event,
                user_id=body.db_user_id,
                anon_id=body.user_id,
            )
    except Exception:
        log.exception("interact failed")
    return {"ok": True}


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
