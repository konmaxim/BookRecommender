

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


# data error: `central` is a strength label that leaked into stance (33 rows).
STANCE_MAP: dict[str, float] = {
    "argues_for":          +1.0,
    "argues_against":      -1.0,
    "presents_both_sides":  0.0,
    "depicts_neutrally":    0.0,
    "not_a_stance_idea":    0.0,
    "central":              0.0,
}

# data error: `substance` is a typo for `substantial` (1 row).
STRENGTH_MAP: dict[str, float] = {
    "central":     1.0,
    "substantial": 0.6,
    "passing":     0.3,
    "substance":   0.6,
}


def load_extractions(
    json_path: Path | str,
    canonical_of: Callable[[int, int], int],
) -> dict[int, dict]:
    """
    Reads the idea-extraction JSON and returns:
        { book_id: {"title": str, "author": str, "ideas": [{"cid", "text", "stance", "strength"}]} }

    `canonical_of(book_id, idea_index) -> canonical_idea_id` joins each
    extracted idea instance to its dedup cluster id. 
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for book_id_str, rec in raw.items():
        book_id = int(book_id_str)
        ideas = []
        for idx, it in enumerate(rec["ideas"]):
            stance_str = it["stance"]
            strength_str = it["strength"]
            if stance_str not in STANCE_MAP:
                raise ValueError(
                    f"unknown stance {stance_str!r} at (book_id={book_id}, idx={idx})"
                )
            if strength_str not in STRENGTH_MAP:
                raise ValueError(
                    f"unknown strength {strength_str!r} at (book_id={book_id}, idx={idx})"
                )
            ideas.append({
                "cid":      canonical_of(book_id, idx),
                "text":     it["idea"],
                "stance":   STANCE_MAP[stance_str],
                "strength": STRENGTH_MAP[strength_str],
            })
        out[book_id] = {
            "title":  rec.get("title", ""),
            "author": rec.get("author", ""),
            "ideas":  ideas,
        }
    return out
