"""Recursive-descent parser for Boolean expressions and quoted phrases."""

from dataclasses import dataclass
import re
from typing import Protocol
from .inverted import DocMeta, Posting

from .tokenizer import tokenize


class EvaluatableIndex(Protocol):
    @property
    def documents(self) -> dict[int, DocMeta]: ...
    @property
    def positions(self) -> dict[str, dict[int, list[int]]] | None: ...

    def __getitem__(self, term: str) -> tuple[Posting, ...]: ...
    def __contains__(self, term: object) -> bool: ...


class Node:
    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    word: str

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        return {p.doc_id for p in index[self.word]} if self.word in index else set()

    def positive_terms(self) -> tuple[str, ...]:
        return (self.word,)


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        if index.positions is None:
            raise ValueError("Phrase search requires an index built with --positions")
        if not self.words:
            return set()
        occurrences = [index.positions.get(word, {}) for word in self.words]
        candidates = set(occurrences[0])
        for items in occurrences[1:]:
            candidates &= set(items)
        return {doc for doc in candidates if any(
            all(start + offset in occurrences[offset][doc] for offset in range(1, len(occurrences)))
            for start in occurrences[0][doc]
        )}

    def positive_terms(self) -> tuple[str, ...]:
        return self.words


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: EvaluatableIndex) -> set[int]:
        return set(index.documents) - self.child.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return ()


LEXER = re.compile(r'"[^"\n]*"|\(|\)|[^\s()"]+')


def _tokens(query: str) -> list[object]:
    result: list[object] = []
    if query.count('"') % 2:
        raise ValueError("Unclosed quotation mark")
    for item in LEXER.findall(query):
        if item in {"(", ")"}:
            result.append(item)
        elif item.startswith('"'):
            words = tuple(tokenize(item[1:-1]))
            if not words:
                raise ValueError("Empty phrase")
            result.append(Phrase(words))
        elif item.upper() in {"AND", "OR", "NOT"} and item.isascii():
            result.append(item.upper())
        else:
            words = list(tokenize(item))
            result.extend(Term(word) for word in words)
    if not result:
        raise ValueError("Empty query")
    # Adjacent operands or ) ( or ) NOT mean implicit AND.
    expanded: list[object] = []
    for token in result:
        if expanded and (isinstance(expanded[-1], Node) or expanded[-1] == ")") and (
            isinstance(token, Node) or token in {"(", "NOT"}
        ):
            expanded.append("AND")
        expanded.append(token)
    return expanded


class Parser:
    def __init__(self, tokens: list[object]):
        self.tokens = tokens
        self.cursor = 0

    def peek(self):
        return self.tokens[self.cursor] if self.cursor < len(self.tokens) else None

    def take(self):
        result = self.peek()
        self.cursor += 1
        return result

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.peek() == "AND":
            self.take()
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        token = self.take()
        if token == "(":
            inner = self.parse_or()
            if self.take() != ")":
                raise ValueError("Unbalanced parentheses")
            return inner
        if isinstance(token, Node):
            return token
        raise ValueError("Expected term or phrase")


def parse_query(query: str) -> Node:
    parser = Parser(_tokens(query))
    node = parser.parse_or()
    if parser.peek() is not None:
        raise ValueError("Unexpected token or unmatched parenthesis")
    return node
