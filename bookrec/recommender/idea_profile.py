

from __future__ import annotations


def build_user_idea_profile(
    saved_book_ids: list[int],
    book_ideas: dict[int, list[dict]],
) -> list[dict]:
    eng: dict[int, float] = {}
    num: dict[int, float] = {}
    den: dict[int, float] = {}

    for bid in saved_book_ids:
        ideas = book_ideas.get(bid)
        if not ideas:
            continue
        for it in ideas:
            cid = int(it["cid"])
            stance = float(it["stance"])
            strength = float(it["strength"])
            eng[cid] = eng.get(cid, 0.0) + strength
            if stance != 0.0:
                num[cid] = num.get(cid, 0.0) + strength * stance
                den[cid] = den.get(cid, 0.0) + strength

    profile: list[dict] = []
    for cid, a in eng.items():
        d = den.get(cid, 0.0)
        p = (num[cid] / d) if d > 0 else 0.0
        if a > 0.0:
            profile.append({"cid": cid, "a": a, "p": p})
    return profile
