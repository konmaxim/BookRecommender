"""
Check pairwise cosine similarity between ALL_THEMES embeddings.

Usage:
    python scripts/theme_orthogonality.py
    python scripts/theme_orthogonality.py --threshold 0.85   # flag pairs above this
    python scripts/theme_orthogonality.py --save heatmap.png
"""

# С помощью этого скрипта определяю, насколько уникальны темы, и какие темы стоит убрать или поменять 
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from bookrec.config import cfg
from bookrec.pipeline.classify import ALL_THEMES


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.where(norms == 0, 1, norms)
    return normed @ normed.T


def main() -> None:
    ap = argparse.ArgumentParser(description="Theme orthogonality check via cosine similarity")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Flag pairs with similarity above this value (default: 0.90)")
    ap.add_argument("--top", type=int, default=20,
                    help="Print this many most-similar pairs (default: 20)")
    ap.add_argument("--save", metavar="FILE", default=None,
                    help="Save heatmap to this file (e.g. heatmap.png)")
    ap.add_argument("--model", default=cfg.embed_model,
                    help=f"Sentence-transformer model to use (default: {cfg.embed_model})")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print(f"Embedding {len(ALL_THEMES)} themes...")
    embeddings = model.encode(ALL_THEMES, show_progress_bar=True, normalize_embeddings=True)
    sim = cosine_similarity_matrix(np.array(embeddings))

    n = len(ALL_THEMES)
    # collect upper-triangle pairs (excluding self-similarity diagonal)
    pairs = [
        (sim[i, j], i, j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    pairs.sort(reverse=True)

    # ── Most similar pairs ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"Top {args.top} most similar theme pairs")
    print(f"{'─'*70}")
    for score, i, j in pairs[: args.top]:
        marker = " ⚠" if score >= args.threshold else ""
        print(f"  {score:.3f}{marker}  [{i:02d}] {ALL_THEMES[i]}")
        print(f"        [{j:02d}] {ALL_THEMES[j]}")
        print()

    # ── Pairs above threshold ────────────────────────────────────────────────
    flagged = [(s, i, j) for s, i, j in pairs if s >= args.threshold]
    print(f"{'─'*70}")
    print(f"Pairs with similarity >= {args.threshold}: {len(flagged)}")
    if flagged:
        print("  Consider merging or removing these themes:")
        for score, i, j in flagged:
            print(f"  {score:.3f}  \"{ALL_THEMES[i]}\"  ↔  \"{ALL_THEMES[j]}\"")

    # ── Per-theme stats ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Per-theme: mean similarity to all other themes (higher = less orthogonal)")
    print(f"{'─'*70}")
    np.fill_diagonal(sim, 0)          # exclude self
    mean_sim = sim.sum(axis=1) / (n - 1)
    order = np.argsort(mean_sim)[::-1]
    for rank, idx in enumerate(order, 1):
        bar = "█" * int(mean_sim[idx] * 40)
        print(f"  {rank:2d}. {mean_sim[idx]:.3f}  {bar}  {ALL_THEMES[idx]}")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    if args.save:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            np.fill_diagonal(sim, 1)  # restore for display
            fig, ax = plt.subplots(figsize=(max(14, n // 3), max(12, n // 3)))
            im = ax.imshow(sim, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            short = [t[:35] for t in ALL_THEMES]
            ax.set_xticklabels(short, rotation=90, fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.02)
            ax.set_title("Theme pairwise cosine similarity", fontsize=11)
            plt.tight_layout()
            plt.savefig(args.save, dpi=150)
            print(f"\nHeatmap saved to {args.save}")
        except ImportError:
            print("\nInstall matplotlib to save the heatmap:  pip install matplotlib")


if __name__ == "__main__":
    main()
