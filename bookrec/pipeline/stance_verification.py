from __future__ import annotations

import json
import logging
from pathlib import Path

from anthropic import Anthropic
from tqdm import tqdm

from ..config import cfg
from ..db import get_db

log = logging.getLogger(__name__)

client = Anthropic(api_key=cfg.anthropic_api_key)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "opus_stance_verification.txt"
VERIFICATION_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")

_VERIFIABLE_STANCES = {"argues_for", "argues_against"}


def _load_descriptions(book_ids: list[int]) -> dict[int, str]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, description FROM books WHERE id = ANY(%s)",
                (book_ids,),
            )
            return {r[0]: r[1] or "" for r in cur.fetchall()}


def verify_stances(title: str, author: str, description: str, ideas: list[dict]) -> list[dict]:
    stance_ideas = [i for i in ideas if i.get("stance") in _VERIFIABLE_STANCES]
    if not stance_ideas:
        return stance_ideas

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        system=[
            {
                "type": "text",
                "text": VERIFICATION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Title: {title}\n"
                    f"Author: {author}\n"
                    f"Description:\n{description}\n\n"
                    f"Ideas to verify:\n{json.dumps(stance_ideas, ensure_ascii=False, indent=2)}\n\n"
                    f"Return ONLY a JSON array in exactly this format, no commentary:\n"
                    f'[{{"idea": "<exact idea text>", "stance": "<corrected stance>"}}]'
                ),
            }
        ],
    )
    #Text cleanup for Opus, errs w/o it 
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text[text.index("\n") + 1:]
        text = text.rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else next(iter(parsed.values()))


def run(ideas_path: str = "data/ideas_validation_haiku_prompt_v1.json") -> None:
    path = Path(ideas_path)
    ideas_data: dict[int, dict] = {
        int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
    }

    to_verify = {
        bid: data for bid, data in ideas_data.items()
        if not data.get("stance_verified")
        and any(i.get("stance") in _VERIFIABLE_STANCES for i in data.get("ideas", []))
    }
    log.info("%d books to verify", len(to_verify))

    descriptions = _load_descriptions(list(to_verify.keys()))

    for book_id, data in tqdm(to_verify.items(), desc="verifying stances"):
        try:
            corrected = verify_stances(
                title=data["title"],
                author=data.get("author", ""),
                description=descriptions.get(book_id, ""),
                ideas=data["ideas"],
            )
            # Replace stances in-place 
            corrected_by_idea = {c["idea"]: c["stance"] for c in corrected}
            for idea in ideas_data[book_id]["ideas"]:
                if idea["idea"] in corrected_by_idea:
                    idea["stance"] = corrected_by_idea[idea["idea"]]
            ideas_data[book_id]["stance_verified"] = True

            path.write_text(json.dumps(ideas_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed for '%s' (id=%d)", data["title"], book_id)

    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run()
