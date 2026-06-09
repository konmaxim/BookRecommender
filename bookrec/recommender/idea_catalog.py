

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .idea_normalize import load_extractions

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
JSON_PATH        = DATA_DIR / "ideas_validation_haiku_prompt_v1.json"
IDEA_TO_CANON    = DATA_DIR / "idea_to_canonical.csv"
CANONICAL_IDEAS  = DATA_DIR / "canonical_ideas.csv"
CENTROIDS_PATH   = DATA_DIR / "canonical_idea_centroids.npy"


def load_canonical_centroids(path: Path = CENTROIDS_PATH) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_canonical_labels(path: Path = CANONICAL_IDEAS) -> dict[int, str]:
    df = pd.read_csv(path)
    return dict(zip(df["canonical_idea_id"].astype(int), df["medoid_text"].astype(str)))


def load_book_ideas(
    json_path: Path = JSON_PATH,
    idea_to_canon_path: Path = IDEA_TO_CANON,
) -> dict[int, list[dict]]:
    
    mapping_df = pd.read_csv(idea_to_canon_path)
    canon_lookup: dict[tuple[int, int], int] = {
        (int(r.book_id), int(r.idea_index)): int(r.canonical_idea_id)
        for r in mapping_df.itertuples(index=False)
    }

    def canonical_of(book_id: int, idea_index: int) -> int:
        try:
            return canon_lookup[(book_id, idea_index)]
        except KeyError as e:
            raise KeyError(
                f"no canonical id for (book_id={book_id}, idea_index={idea_index}); "
                "rerun bookrec.pipeline.idea_clustering after extracting new ideas"
            ) from e

    books = load_extractions(json_path, canonical_of)
    out: dict[int, list[dict]] = {bid: rec["ideas"] for bid, rec in books.items()}
    log.info("loaded %d books with idea instances", len(out))
    return out
