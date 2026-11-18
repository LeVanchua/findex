"""Unicode-aware streaming tokenizer.

Policy: Unicode NFKC + casefold; combine accents are preserved; curly
apostrophe is unified to ASCII apostrophe. Apostrophes inside words are kept
(п'ять); leading/trailing apostrophes are separators. Hyphens and dashes
split words, including inside compound words. Decimal digits remain tokens,
and digit/letter mixtures are one token (python3); punctuation and emoji split.
"""

import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens one at a time without building a token list."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'").replace("ʼ", "'").replace("`", "'")
    # Regex \w excludes combining marks; NFC composes most normal accents.
    normalized = unicodedata.normalize("NFC", normalized)
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group()
