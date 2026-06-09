from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EMBEDDINGS_IN = DATA_DIR / "idea_embeddings.npy"
METADATA_IN   = DATA_DIR / "idea_metadata.csv"

IDEA_TO_CANONICAL_OUT = DATA_DIR / "idea_to_canonical.csv"
CANONICAL_IDEAS_OUT   = DATA_DIR / "canonical_ideas.csv"
CENTROIDS_OUT         = DATA_DIR / "canonical_idea_centroids.npy"

DISTANCE_THRESHOLD = 0.15  


def run(distance_threshold: float = DISTANCE_THRESHOLD) -> None:
    embeddings = np.load(EMBEDDINGS_IN).astype(np.float32)
    meta = pd.read_csv(METADATA_IN)
    if len(meta) != len(embeddings):
        raise ValueError(f"row mismatch: {len(meta)} metadata vs {len(embeddings)} embeddings")
    log.info("loaded %d ideas, dim=%d", len(meta), embeddings.shape[1])

    # idea_index within each book preserves JSON order 
    meta = meta.reset_index(drop=True)
    meta["idea_index"] = meta.groupby("book_id").cumcount()

    log.info("clustering (complete linkage, cosine, threshold=%s)…", distance_threshold)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="complete",
        distance_threshold=distance_threshold,
    )
    labels = clusterer.fit_predict(embeddings)
    n_canonical = int(labels.max()) + 1
    log.info("found %d canonical ideas from %d raw", n_canonical, len(labels))

    centroids = np.zeros((n_canonical, embeddings.shape[1]), dtype=np.float32)
    sizes = np.zeros(n_canonical, dtype=np.int64)
    for row_idx, cid in enumerate(labels):
        centroids[cid] += embeddings[row_idx]
        sizes[cid] += 1
    centroids /= sizes[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids /= norms

    # medoid: member whose embedding has max cosine to the centroid
    medoid_text = [""] * n_canonical
    cosines = embeddings @ centroids.T  # (N, n_canonical)
    best_row_per_cluster: dict[int, tuple[float, int]] = {}
    for row_idx, cid in enumerate(labels):
        score = float(cosines[row_idx, cid])
        prev = best_row_per_cluster.get(cid)
        if prev is None or score > prev[0]:
            best_row_per_cluster[cid] = (score, row_idx)
    for cid, (_, row_idx) in best_row_per_cluster.items():
        medoid_text[cid] = str(meta.iloc[row_idx]["idea"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    idea_to_canonical = meta[["book_id", "idea_index"]].copy()
    idea_to_canonical["canonical_idea_id"] = labels.astype(np.int32)
    idea_to_canonical.to_csv(IDEA_TO_CANONICAL_OUT, index=False)
    log.info("wrote %s (%d rows)", IDEA_TO_CANONICAL_OUT, len(idea_to_canonical))

    canonical = pd.DataFrame({
        "canonical_idea_id": np.arange(n_canonical, dtype=np.int32),
        "medoid_text": medoid_text,
        "size": sizes,
    })
    canonical.to_csv(CANONICAL_IDEAS_OUT, index=False)
    log.info("wrote %s (%d rows)", CANONICAL_IDEAS_OUT, len(canonical))

    np.save(CENTROIDS_OUT, centroids)
    log.info("wrote %s (shape=%s)", CENTROIDS_OUT, centroids.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run()
