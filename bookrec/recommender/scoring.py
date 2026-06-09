from __future__ import annotations

import numpy as np

from .idea_scoring import score_ideas as _score_ideas, blend


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_themes(
    user_weights: dict[str, float],
    book_themes: dict[str, float],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
) -> float:
    """Theme channel — extracted from the old score_one_book body."""
    total = 0.0
    prominence_sum = sum(p for t, p in book_themes.items() if t in theme_index)

    for user_theme, weight in user_weights.items():
        if user_theme not in theme_index:
            continue
        u_vec = theme_matrix[theme_index[user_theme]]
        best = 0.0
        for book_theme, prominence in book_themes.items():
            if book_theme not in theme_index:
                continue
            b_vec = theme_matrix[theme_index[book_theme]]
            similarity = _cosine(u_vec, b_vec)
            best = max(best, similarity * prominence)
        total += weight * best

    return total / max(1.0, prominence_sum ** 0.5)


def score_one_book(
    user_weights: dict[str, float],
    book_themes: dict[str, float],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
    *,
    idea_profile: list[dict] | None = None,
    candidate_ideas: list[dict] | None = None,
    centroids: np.ndarray | None = None,
    gamma: float = 1.0,
    lam: float = 2.0,
    beta: float = 0.4,
    sim_floor: float = 0.5,
) -> float:
    """
    Combined per-book score. The idea channel is skipped when its inputs
    are absent (e.g. user has no profile yet) — returns just the theme term.
    """
    theme = score_themes(user_weights, book_themes, theme_matrix, theme_index)
    if not idea_profile or not candidate_ideas or centroids is None:
        return theme
    idea = _score_ideas(
        idea_profile,
        candidate_ideas,
        centroids,
        lam=lam,
        beta=beta,
        sim_floor=sim_floor,
    )
    return blend(theme, idea, gamma)


def score_all_books(
    user_weights: dict[str, float],
    books_themes: dict[int, dict[str, float]],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
    *,
    idea_profile: list[dict] | None = None,
    book_ideas: dict[int, list[dict]] | None = None,
    centroids: np.ndarray | None = None,
    gamma: float = 1.0,
    lam: float = 2.0,
    beta: float = 0.4,
    sim_floor: float = 0.5,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for bid, book_themes in books_themes.items():
        candidate_ideas = book_ideas.get(bid) if book_ideas is not None else None
        out[bid] = score_one_book(
            user_weights,
            book_themes,
            theme_matrix,
            theme_index,
            idea_profile=idea_profile,
            candidate_ideas=candidate_ideas,
            centroids=centroids,
            gamma=gamma,
            lam=lam,
            beta=beta,
            sim_floor=sim_floor,
        )
    return out
