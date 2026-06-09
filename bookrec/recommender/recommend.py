from __future__ import annotations

import numpy as np

from .profile import compute_user_mass, compute_user_weights
from .scoring import score_all_books


def recommend(
    user_book_ids: list[int],
    books_themes: dict[int, dict[str, float]],
    catalog_mass: dict[str, float],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
    excluded_ids: set[int] | None = None,
    top_k: int = 20,
    *,
    idea_profile: list[dict] | None = None,
    book_ideas: dict[int, list[dict]] | None = None,
    centroids: np.ndarray | None = None,
    gamma: float = 1.0,
    lam: float = 2.0,
    beta: float = 0.4,
    sim_floor: float = 0.5,
) -> list[tuple[int, float]]:

    if not user_book_ids:
        return []

    user_mass = compute_user_mass(user_book_ids, books_themes)
    if not user_mass:
        return []

    user_weights = compute_user_weights(user_mass, catalog_mass)

    seen = set(user_book_ids) | (excluded_ids or set())
    candidates = {bid: themes for bid, themes in books_themes.items() if bid not in seen}
    if not candidates:
        return []

    scores = score_all_books(
        user_weights,
        candidates,
        theme_matrix,
        theme_index,
        idea_profile=idea_profile,
        book_ideas=book_ideas,
        centroids=centroids,
        gamma=gamma,
        lam=lam,
        beta=beta,
        sim_floor=sim_floor,
    )
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
