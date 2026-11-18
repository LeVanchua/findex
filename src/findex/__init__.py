"""A small streaming text processing toolkit."""

from .corpus import Document, iter_documents
from .tokenizer import tokenize

__all__ = ["Document", "iter_documents", "tokenize"]
