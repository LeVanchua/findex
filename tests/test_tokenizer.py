from findex.tokenizer import tokenize


def test_casefold_and_nfkc():
    assert list(tokenize("ＰＹＴＨＯＮ Python УКРАЇНА")) == ["python", "python", "україна"]


def test_apostrophes():
    assert list(tokenize("П’ять п'ять обʼєкт 'текст'")) == ["п'ять", "п'ять", "об'єкт", "текст"]


def test_hyphens_numbers_and_underscores():
    assert list(tokenize("онлайн-пошук 2026 python3 слово_слово 3.14")) == [
        "онлайн", "пошук", "2026", "python3", "слово", "слово", "3", "14"
    ]


def test_empty_punctuation_and_emoji():
    assert list(tokenize("  !!! 🤖  ")) == []


def test_generator_is_lazy():
    result = tokenize("Один два")
    assert iter(result) is result
    assert next(result) == "один"
    assert list(result) == ["два"]
