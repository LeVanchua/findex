"""Boolean query parser and sorted-list merge/set execution engines.

Grammar: NOT > AND > OR; adjacency means AND; parentheses supported.
"""

import re

from .inverted import InvertedIndex
from .tokenizer import tokenize


def merge_and(left: list[int], right: list[int]) -> list[int]:
    i = j = 0
    result: list[int] = []
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(left: list[int], right: list[int]) -> list[int]:
    i = j = 0
    result: list[int] = []
    while i < len(left) or j < len(right):
        if j >= len(right) or (i < len(left) and left[i] < right[j]):
            value = left[i]
            i += 1
        elif i >= len(left) or right[j] < left[i]:
            value = right[j]
            j += 1
        else:
            value = left[i]
            i += 1
            j += 1
        result.append(value)
    return result


def merge_not(universe: list[int], excluded: list[int]) -> list[int]:
    result: list[int] = []
    j = 0
    for doc_id in universe:
        while j < len(excluded) and excluded[j] < doc_id:
            j += 1
        if j == len(excluded) or excluded[j] != doc_id:
            result.append(doc_id)
    return result


def _lex(query: str) -> list[str]:
    chunks = re.findall(r"\(|\)|[^\s()]+", query)
    result: list[str] = []
    for chunk in chunks:
        if chunk in {"(", ")"}:
            result.append(chunk)
        elif chunk.upper() in {"AND", "OR", "NOT"} and chunk.isascii():
            result.append(chunk.upper())
        else:
            result.extend(tokenize(chunk))
    if not result:
        raise ValueError("Empty query")
    # Insert implicit AND (between terms, closing/opening brackets and NOT).
    expanded: list[str] = []
    for token in result:
        if expanded and (expanded[-1] not in {"AND", "OR", "NOT", "("}) and token not in {"AND", "OR", ")"}:
            expanded.append("AND")
        expanded.append(token)
    return expanded


def _rpn(query: str) -> list[str]:
    priority = {"OR": 1, "AND": 2, "NOT": 3}
    output: list[str] = []
    operators: list[str] = []
    for token in _lex(query):
        if token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if not operators:
                raise ValueError("Unbalanced parentheses")
            operators.pop()
        elif token in priority:
            while operators and operators[-1] != "(" and (
                priority[operators[-1]] > priority[token]
                or (priority[operators[-1]] == priority[token] and token != "NOT")
            ):
                output.append(operators.pop())
            operators.append(token)
        else:
            output.append(token)
    while operators:
        token = operators.pop()
        if token == "(":
            raise ValueError("Unbalanced parentheses")
        output.append(token)
    return output


def search(index: InvertedIndex, query: str, engine: str = "merge") -> list[int]:
    if engine not in {"merge", "set"}:
        raise ValueError("engine must be merge or set")
    all_ids = sorted(index.documents)
    stack: list[list[int] | set[int]] = []
    for token in _rpn(query):
        if token == "NOT":
            if not stack:
                raise ValueError("NOT requires an operand")
            value = stack.pop()
            stack.append(merge_not(all_ids, value) if engine == "merge" else set(all_ids) - set(value))
        elif token in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError(f"{token} requires two operands")
            right, left = stack.pop(), stack.pop()
            if engine == "merge":
                stack.append(merge_and(left, right) if token == "AND" else merge_or(left, right))
            else:
                stack.append(left & right if token == "AND" else left | right)
        else:
            docs = [p.doc_id for p in index.postings.get(token, [])]
            stack.append(docs if engine == "merge" else set(docs))
    if len(stack) != 1:
        raise ValueError("Malformed query")
    return sorted(stack[0])
