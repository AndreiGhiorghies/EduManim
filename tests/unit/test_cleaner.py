from tools.tts.cleaner import clean_for_tts, split_sentences


def test_empty_string_returns_empty():
    assert clean_for_tts("") == ""


def test_sum_symbol():
    assert "the sum of" in clean_for_tts(r"\sum_{i=1}^{n}")


def test_greek_letter_alpha():
    assert clean_for_tts(r"\alpha + \beta") == "alpha + beta"


def test_fraction():
    result = clean_for_tts(r"\frac{a}{b}")
    assert "a over b" in result


def test_square_root():
    result = clean_for_tts(r"\sqrt{x}")
    assert "the square root of x" in result


def test_superscript_brace():
    result = clean_for_tts("x^{2}")
    assert "to the power of 2" in result


def test_superscript_single_char():
    result = clean_for_tts("x^2")
    assert "to the power of 2" in result


def test_subscript():
    result = clean_for_tts("x_{i}")
    assert "sub i" in result


def test_math_delimiters_stripped():
    result = clean_for_tts("$x + y$")
    assert "$" not in result


def test_unknown_command_drops_backslash_keeps_word():
    result = clean_for_tts(r"\unknowncommand")
    assert "\\" not in result
    assert "unknowncommand" in result


def test_whitespace_normalized():
    result = clean_for_tts("a    b\n\nc")
    assert result == "a b c"


def test_full_pythagorean_example():
    result = clean_for_tts("a^2 + b^2 = c^2")
    assert "to the power of 2" in result
    assert result.count("to the power of 2") == 3


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_sentences_single_short_sentence():
    assert split_sentences("Hello world.") == ["Hello world."]


def test_split_sentences_groups_under_max_chars():
    text = "One. Two. Three."
    chunks = split_sentences(text, max_chars=220)
    assert chunks == ["One. Two. Three."]


def test_split_sentences_splits_when_over_max_chars():
    text = "This is sentence one. This is sentence two."
    chunks = split_sentences(text, max_chars=25)
    assert len(chunks) > 1
    assert all(len(c) <= 25 or " " not in c for c in chunks)
