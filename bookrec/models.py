from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserRating:
    book_id: int
    rating: int                  # 1-5
    book_title: Optional[str] = None
    book_url: Optional[str] = None


@dataclass
class User:
    name: str
    id: Optional[int] = None
    ratings: list[UserRating] = field(default_factory=list)


@dataclass
class Book:
    title: str
    author: str
    book_url: str
    genre: str
    rating: Optional[float] = None
    isbn: Optional[str] = None
    year: Optional[int] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    readers_count: Optional[int] = None
    reviews_count: Optional[int] = None
    quotes_count: Optional[int] = None
    description: Optional[str] = None
