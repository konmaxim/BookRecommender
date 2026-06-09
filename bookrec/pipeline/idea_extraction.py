from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from psycopg2.extras import Json
from tqdm import tqdm

from ..config import cfg
from ..db import get_db

log = logging.getLogger(__name__)

client = Anthropic(api_key=cfg.anthropic_api_key)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "idea_extraction_prompt_v1.txt"
EXTRACTION_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")


def extract_ideas(title: str, author: str, description: str) -> list[dict]:
    response = client.messages.create(
        model=cfg.claude_model,
        max_tokens=4000,
        system=[
            {
                "type": "text",
                "text": EXTRACTION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"BOOK TO ANALYZE\n\nTitle: {title}\nAuthor: {author}\nDescription:\n{description}",
            },
        ],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]          # cleanup
        text = text[text.index("\n") + 1:]       # remove json
        text = text.rsplit("```", 1)[0].strip()  
    return json.loads(text)["ideas"]


VALIDATION_TITLES = [
    "Что делать?",
    "Атлант расправил плечи",
    "Братья Карамазовы", 
    "Воскресение",
    "Мартин Иден",
    "Норвежский Лес",
    "От Руси до России",
    "Илиада",
    "Евгений Онегин. Роман в стихах",
    "Тихий Дон",
    "Дядя Ваня",
    "Капитал",
    "Критика чистого разума (с комментариями и иллюстрациями)",
    "Мастер и Маргарита",
]


def extract_books(titles: list[str] | None = VALIDATION_TITLES) -> pd.DataFrame:
    with get_db() as conn:
        with conn.cursor() as cur:
            if titles is None:
                cur.execute("""
                    SELECT DISTINCT ON (title) id, title, author, genre, rating, year, description, tags
                    FROM books
                    WHERE description IS NOT NULL AND description != ''
                      AND genre != 'Программирование'
                    ORDER BY title, id
                """)
            else:
                cur.execute("""
                    SELECT id, title, author, genre, rating, year, description, tags
                    FROM books
                    WHERE title = ANY(%s)
                    ORDER BY title
                """, (titles,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def run(output_path: str = "data/ideas_validation_haiku_prompt_v1.json") -> None:
    df = extract_books(titles=None)
    log.info("%d books loaded", len(df))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        results: dict[int, dict] = {int(k): v for k, v in json.loads(out.read_text(encoding="utf-8")).items()}
        log.info("Resuming — %d already done", len(results))
    else:
        results = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting ideas"):
        book_id = int(row["id"])
        if book_id in results:
            continue
        try:
            ideas = extract_ideas(
                title=row["title"],
                author=row["author"] or "",
                description=row["description"] or "",
            )
            results[book_id] = {
                "title": row["title"],
                "author": row["author"] or "",
                "ideas": ideas,
            }
            out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed for '%s' (id=%d)", row["title"], book_id)

    log.info("Done. %d / %d books saved to %s", len(results), len(df), out)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run()
