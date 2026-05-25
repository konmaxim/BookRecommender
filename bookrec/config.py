from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DBConfig:
    host: str = field(default_factory=lambda: os.getenv("PGHOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("PGPORT", "5432")))
    dbname: str = field(default_factory=lambda: os.getenv("PGDATABASE", ""))
    user: str = field(default_factory=lambda: os.getenv("PGUSER", ""))
    password: str = field(default_factory=lambda: os.getenv("PGPASSWORD", ""))

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
        }


@dataclass
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Scraper
    scrape_delay_min: float = 0.5
    scrape_delay_max: float = 1.5
    scrape_long_break_every: int = 20
    scrape_rewarm_every: int = 40
    livelib_cookies: str = field(default_factory=lambda: os.getenv("LIVELIB_COOKIES", ""))
    # BERTopic
    bertopic_min_topic_size: int = 4

    # Models
    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    classify_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    claude_model: str = "claude-sonnet-4-6"

    # Thresholds
    theme_score_threshold: float = 0.3
    embed_batch_size: int = 32

    # Recommender
    theme_length_norm_alpha: float = 0.75  # 0 = no per-book length effect, 1 = full L2 norm


cfg = Config()
