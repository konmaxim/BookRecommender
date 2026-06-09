"""

Usage:
    bookrec scrape-books     [--pages N] [--genres NAME ...]
    bookrec scrape-reviews   [--pages N] [--limit N]
    bookrec classify         [--zeroshot]
    bookrec themes
    bookrec embed            [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )




def _cmd_scrape_books(args: argparse.Namespace) -> None:
    from . import db
    from .scraper import books as book_scraper

    with db.get_db() as conn:
        db.create_tables(conn)
        book_scraper.run(
            conn,
            pages=args.pages,
            genres=args.genres or None,
            all_genres=args.all_genres,
        )


def _cmd_scrape_reviews(args: argparse.Namespace) -> None:
    from . import db
    from .scraper import reviews as review_scraper

    with db.get_db() as conn:
        db.create_tables(conn)
        review_scraper.run(conn, pages=args.pages, limit=args.limit, cookies=args.cookies)


def _cmd_scrape_top_reviews(args: argparse.Namespace) -> None:
    from . import db
    from .scraper import top_reviews as top_review_scraper

    with db.get_db() as conn:
        top_review_scraper.run(conn, limit=args.limit, dry_run=args.dry_run, cookies=args.cookies)


def _cmd_classify(args: argparse.Namespace) -> None:
    from . import db
    from .pipeline import classify

    with db.get_db() as conn:
        classify.run(conn, only_bertopic=args.only_bertopic)


def _cmd_themes(args: argparse.Namespace) -> None:
    from . import db
    from .pipeline import themes

    with db.get_db() as conn:
        themes.run(conn)


def _cmd_embed(args: argparse.Namespace) -> None:
    from . import db
    from .pipeline import embed

    with db.get_db() as conn:
        embed.run(conn, output_dir=args.output_dir)


def _cmd_cluster_ideas(args: argparse.Namespace) -> None:
    from .pipeline import idea_clustering
    idea_clustering.run(distance_threshold=args.distance_threshold)




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookrec",
        description="BookRecommendor — livelib.ru scraping and NLP pipeline",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # scrape-books
    p_books = sub.add_parser("scrape-books", help="Scrape book listings from livelib.ru")
    p_books.add_argument("--pages", type=int, default=4, help="Pages per genre (default: 4)")
    p_books.add_argument("--genres", nargs="+", metavar="GENRE", help="Restrict to these genre names")
    p_books.add_argument(
        "--all-genres",
        action="store_true",
        help="Include already-scraped genres (default: only new genres)",
    )
    p_books.set_defaults(func=_cmd_scrape_books)

    # scrape-reviews
    p_reviews = sub.add_parser("scrape-reviews", help="Scrape user reviews for all books in DB")
    p_reviews.add_argument("--pages", type=int, default=5, help="Max review pages per book (default: 5)")
    p_reviews.add_argument("--limit", type=int, default=None, help="Max number of books to process")
    p_reviews.add_argument("--cookies", metavar="COOKIE_STRING", default=None, help="Raw Cookie header from browser (overrides LIVELIB_COOKIES env var)")
    p_reviews.set_defaults(func=_cmd_scrape_reviews)

    # scrape-top-reviews
    p_top = sub.add_parser(
        "scrape-top-reviews",
        help="Scrape top-4 reviews (by likes) for books not yet zero-shot classified",
    )
    p_top.add_argument("--limit", type=int, default=None, help="Max books to process")
    p_top.add_argument("--dry-run", action="store_true", help="Print results without saving to DB")
    p_top.add_argument("--cookies", metavar="COOKIE_STRING", default=None, help="Raw Cookie header from browser (overrides LIVELIB_COOKIES env var)")
    p_top.set_defaults(func=_cmd_scrape_top_reviews)

    # classify
    p_classify = sub.add_parser(
        "classify",
        help="Two-stage classification: zero-shot then BERTopic+Claude fallback",
    )
    p_classify.add_argument(
        "--only-bertopic",
        action="store_true",
        help="Re-run BERTopic only on books already classified by BERTopic (skips zero-shot)",
    )
    p_classify.set_defaults(func=_cmd_classify)

    # themes
    p_themes = sub.add_parser("themes", help="Zero-shot classification against predefined themes")
    p_themes.set_defaults(func=_cmd_themes)

    # embed
    p_embed = sub.add_parser("embed", help="Generate sentence embeddings for reviews")
    p_embed.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write embeddings and metadata (default: current dir)",
    )
    p_embed.set_defaults(func=_cmd_embed)

    # cluster-ideas
    p_cluster = sub.add_parser(
        "cluster-ideas",
        help="Build canonical idea vocabulary via complete-linkage clustering of FRIDA embeddings",
    )
    p_cluster.add_argument(
        "--distance-threshold",
        type=float,
        default=0.1,
        help="Cosine distance threshold for cluster merge (default: 0.1 = similarity 0.9)",
    )
    p_cluster.set_defaults(func=_cmd_cluster_ideas)

    return parser



def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
