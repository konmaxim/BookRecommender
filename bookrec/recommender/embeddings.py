from __future__ import annotations
import logging
import numpy as np
from ..config import cfg
from ..pipeline.classify import ALL_THEMES

log = logging.getLogger(__name__)

#Загружает эмбеддинги тем 

class ThemeEmbeddings:

    def __init__(self) -> None:
        self._themes: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None  # shape (N, d)

    def build(self, model_name: str = cfg.embed_model) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Building theme embeddings with %s ...", model_name)
        model = SentenceTransformer(model_name)
        vecs = model.encode(
            ALL_THEMES,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        self._themes = list(ALL_THEMES)
        self._index = {t: i for i, t in enumerate(self._themes)}
        self._matrix = np.array(vecs, dtype=np.float32)
        log.info("%d themes × %d dims", len(self._themes), self._matrix.shape[1])

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self.build()
        return self._matrix

    @property
    def index(self) -> dict[str, int]:
        if self._matrix is None:
            self.build()
        return self._index

    @property
    def themes(self) -> list[str]:
        if self._matrix is None:
            self.build()
        return self._themes
