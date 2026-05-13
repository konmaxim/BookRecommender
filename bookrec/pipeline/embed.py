from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config import cfg

log = logging.getLogger(__name__)

_QUERY = """
    SELECT
        r.id AS review_id,
        r.book_id,
        b.title,
        b.author,
        b.genre,
        CONCAT('[', b.genre, '] ', r.review_text) AS sentence
    FROM reviews r
    JOIN books b ON r.book_id = b.id
    WHERE r.review_text IS NOT NULL
"""


def run(conn, output_dir: str = ".") -> None:
    """Generate sentence embeddings for all reviews and save to disk.

    Output files:
        <output_dir>/embeddings_rosbert.npy  — float32 embedding matrix (N × D)
        <output_dir>/embeddings_metadata.csv — review_id, book_id, title, author, genre, sentence

    Args:
        conn: Active psycopg2 connection.
        output_dir: Directory to write output files into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading reviews from DB...")
    df = pd.read_sql(_QUERY, conn)
    log.info("Loaded %d reviews", len(df))

    log.info("Loading model: %s", cfg.embed_model)
    model = SentenceTransformer(cfg.embed_model)

    log.info("Encoding...")
    embeddings = model.encode(
        df["sentence"].tolist(),
        show_progress_bar=True,
        batch_size=cfg.embed_batch_size,
    )

    emb_path = out / "embeddings_rosbert.npy"
    meta_path = out / "embeddings_metadata.csv"

    np.save(emb_path, embeddings)
    df[["review_id", "book_id", "title", "author", "genre", "sentence"]].to_csv(
        meta_path, index=False
    )

    log.info("Saved %d embeddings → %s", len(embeddings), emb_path)
    log.info("Metadata → %s", meta_path)
