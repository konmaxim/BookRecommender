from __future__ import annotations

from collections import defaultdict

import numpy as np

#compute how important a user is to a theme: 
#(sum of nli scores of a theme)/(sum of all themes across user books)
def compute_user_mass(
    user_book_ids: list[int],
    books_themes: dict[int, dict[str, float]],
) -> dict[str, float]:
    theme_totals: dict[str, float] = defaultdict(float)
    grand_total = 0.0
    for b_id in user_book_ids:
        if b_id not in books_themes:
            continue
        for theme, score in books_themes[b_id].items():
            theme_totals[theme] += score
            grand_total += score
    if grand_total == 0:
        return {}
    return {t: total / grand_total for t, total in theme_totals.items()}


def compute_user_weights(
    user_mass: dict[str, float],
    catalog_mass: dict[str, float],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for theme, u_mass in user_mass.items():
        c_mass = catalog_mass.get(theme)
        if c_mass is None:
            continue
        weights[theme] = max(-0.5, float(np.log(u_mass / c_mass)))
    return weights
