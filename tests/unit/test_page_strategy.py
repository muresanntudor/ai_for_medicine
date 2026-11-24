import pytest

from rag.adapters.pdf.page_strategy import MIN_CHARACTERS, choose_extraction


def test_empty_page_needs_ocr():
    assert choose_extraction("") == "ocr"


def test_too_few_characters_needs_ocr():
    assert choose_extraction("a" * (MIN_CHARACTERS - 1)) == "ocr"


def test_enough_characters_is_text():
    assert choose_extraction("a" * MIN_CHARACTERS) == "text"


def test_whitespace_does_not_count():
    assert choose_extraction("  \n\n\t   abc   \n" * 10) == "ocr"


def test_normal_page_with_many_lines_is_text():
    page = "Pacientul prezintă dureri toracice.\n" * 30
    assert choose_extraction(page) == "text"


@pytest.mark.parametrize("bad_share, expected", [(0.4, "ocr"), (0.2, "text")])
def test_broken_font_characters(bad_share: float, expected: str):
    bad = int(100 * bad_share)
    page = "�" * bad + "a" * (100 - bad)
    assert choose_extraction(page) == expected


def test_control_characters_count_as_bad():
    assert choose_extraction("\x00\x01\x02" * 20 + "a" * 40) == "ocr"


def test_romanian_diacritics_are_text():
    page = "Inima, ficatul și plămânii sunt organe vitale; țesutul și ăsta. " * 3
    assert choose_extraction(page) == "text"
