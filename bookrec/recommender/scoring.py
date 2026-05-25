from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def score_one_book(
    user_weights: dict[str, float],
    book_themes: dict[str, float],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
) -> float:
    total = 0.0

    for user_theme, weight in user_weights.items():
        #check that user theme is defined in the list of themes
        if user_theme not in theme_index:
            continue
        u_vec = theme_matrix[theme_index[user_theme]]

        #find the best matching book theme 
        best = 0.0
        for book_theme, prominence in book_themes.items():
            if book_theme not in theme_index:
                continue
            b_vec = theme_matrix[theme_index[book_theme]]
            similarity = _cosine(u_vec, b_vec)
            best = max(best, similarity * prominence)

        total += weight * best

    return total


def score_all_books(
    user_weights: dict[str, float],
    books_themes: dict[int, dict[str, float]],
    theme_matrix: np.ndarray,
    theme_index: dict[str, int],
) -> dict[int, float]:
    return {
        bid: score_one_book(user_weights, book_themes, theme_matrix, theme_index)
        for bid, book_themes in books_themes.items()
    }
