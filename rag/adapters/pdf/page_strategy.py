"""
It looks at the text pulled from one page and decides:
is this real text, or is the page basically a picture (a scan) that needs OCR?
"""

from typing import Literal

MIN_CHARACTERS = 50
MAX_BAD_CHARACTERS = 0.3

# What broken fonts produce instead of real letters ("�").
REPLACEMENT_CHARACTER = "�"


def choose_extraction(text: str) -> Literal["text", "ocr"]:
    # Spaces and newlines prove nothing, so only real characters are counted.
    cleaned_text = [char for char in text if not char.isspace()]

    if len(cleaned_text) < MIN_CHARACTERS:
        return "ocr"

    count_bad = 0
    for char in cleaned_text:
        if not char.isprintable() or char == REPLACEMENT_CHARACTER:
            count_bad += 1

    if count_bad / len(cleaned_text) > MAX_BAD_CHARACTERS:
        return "ocr"
    return "text"
