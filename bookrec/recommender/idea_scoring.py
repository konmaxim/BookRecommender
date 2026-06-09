
from __future__ import annotations

import numpy as np


def score_ideas(
    profile: list[dict],
    candidate_ideas: list[dict],
    centroids: np.ndarray,
    *,
    lam: float = 2.0,
    beta: float = 0.4,
    sim_floor: float = 0.5,
) -> float:
    if not profile or not candidate_ideas:
        return 0.0

    cand_cids: dict[int, dict] = {int(d["cid"]): d for d in candidate_ideas}
    cand_cid_list = [int(d["cid"]) for d in candidate_ideas]
    cand_embs = centroids[cand_cid_list]  # (n_cand, dim)

    total = 0.0
    for it in profile:
        a = float(it["a"])
        if a <= 0.0:
            continue
        p = float(it["p"])
        cid_i = int(it["cid"])

        if cid_i in cand_cids:
            m = cand_cids[cid_i]
            sim = 1.0
        else:
            e_i = centroids[cid_i]
            sims = cand_embs @ e_i
            k = int(np.argmax(sims))
            sim = float(sims[k])
            if sim < sim_floor:
                continue
            m = candidate_ideas[k]

        s_c = float(m["stance"])
        sigma_c = float(m["strength"])

        if p != 0.0 and s_c != 0.0:
            base = a * sim * sigma_c * (s_c * p)
            contrib = base if base >= 0.0 else lam * base
        else:
            contrib = beta * a * sim * sigma_c

        total += contrib

    return total


def blend(theme_score: float, idea_score: float, gamma: float) -> float:
    return float(theme_score) + float(gamma) * float(idea_score)
