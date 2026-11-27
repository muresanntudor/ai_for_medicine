"""
Removes the noise the parser leaves in a ParsedDocument, so search only sees real content:
running headers and page numbers, figure callout labels, and words split across lines.

Pipes & Filters: each filter is a small pure function ParsedDocument -> ParsedDocument,
and clean() runs them one after another. The models are frozen, so a filter never
changes the document it gets; it builds a new one with model_copy(update=...).
Imports only rag.domain: by this point there is no PDF library anymore.
"""

import re
from collections import Counter
from collections.abc import Callable

from rag.domain.models import Block, Line, PageText, ParsedDocument, compute_page_offsets

# Header/footer detection only makes sense when there are enough pages to see repetition.
MIN_PAGES_FOR_HEADERS = 3
# A top/bottom line must repeat on at least this many pages to count as header/footer.
MIN_HEADER_REPEATS = 3
# A removed header needs at least this many letters to be kept as the page's section
# (so a lone decorative "A" in the margin is not mistaken for a chapter name).
MIN_SECTION_LETTERS = 3
# Blocks whose every line is this short are figure callouts ("1", "2", "a)").
MAX_LABEL_CHARACTERS = 3

_DIGITS = re.compile(r"\d")

Filter = Callable[[ParsedDocument], ParsedDocument]


def clean(doc: ParsedDocument) -> ParsedDocument:
    """Run every filter in order, then recompute page_offsets for the new text."""
    filters: list[Filter] = [remove_headers_footers, drop_figure_labels, join_hyphenated_lines]
    for step in filters:
        doc = step(doc)
    return doc.model_copy(update={"page_offsets": compute_page_offsets(doc.pages)})


# ---------- filter 1: running headers and page numbers ----------

def remove_headers_footers(doc: ParsedDocument) -> ParsedDocument:
    """Remove lines repeated at the very top or bottom of many pages.

    The running header's text is not thrown away: it is saved on the page as
    `section`, so chunking can later put it in front of each chunk as context.
    Page numbers (no letters) are simply dropped; page_label already has them.
    """
    if len(doc.pages) < MIN_PAGES_FOR_HEADERS:
        return doc

    # 1. On every page, the topmost and bottommost lines are the candidates.
    candidates_per_page = [_edge_lines(page) for page in doc.pages]

    # 2. Count on how many pages each normalised candidate appears (a set: once per page).
    counts: Counter[str] = Counter()
    for candidates in candidates_per_page:
        counts.update({_normalise(line.text) for line in candidates})
    repeated = {text for text, n in counts.items() if n >= MIN_HEADER_REPEATS}

    # 3. Remove the repeated candidates from their page, remembering the header text.
    pages = []
    for page, candidates in zip(doc.pages, candidates_per_page):
        to_remove = [line for line in candidates if _normalise(line.text) in repeated]
        page = _without_lines(page, to_remove)
        section = _section_from(to_remove)
        if section:
            page = page.model_copy(update={"section": section})
        pages.append(page)
    return doc.model_copy(update={"pages": pages})


def _edge_lines(page: PageText) -> list[Line]:
    """The topmost and bottommost line of a page (one line if the page has only one)."""
    lines = [line for block in page.blocks for line in block.lines]
    if not lines:
        return []
    top = min(lines, key=lambda line: line.y_top)
    bottom = max(lines, key=lambda line: line.y_bottom)
    return [top] if top is bottom else [top, bottom]


def _normalise(text: str) -> str:
    """Lowercase and digits -> '#', so page numbers "42" and "43" look the same."""
    return _DIGITS.sub("#", text.strip().lower())


def _section_from(removed: list[Line]) -> str:
    """The running header among the removed lines: the topmost one that has letters."""
    with_letters = [
        line for line in removed
        if sum(ch.isalpha() for ch in line.text) >= MIN_SECTION_LETTERS
    ]
    if not with_letters:
        return ""
    return min(with_letters, key=lambda line: line.y_top).text


def _without_lines(page: PageText, to_remove: list[Line]) -> PageText:
    """A copy of `page` without the given lines; blocks left empty are dropped.

    Lines are compared with `is` (same object), not `==`: two different lines
    with identical text and position must not both disappear.
    """
    blocks: list[Block] = []
    for block in page.blocks:
        kept = [line for line in block.lines if not any(line is gone for gone in to_remove)]
        if kept:
            blocks.append(block.model_copy(update={"lines": kept}))
    return page.model_copy(update={"blocks": blocks})


# ---------- filter 2: figure callout labels ----------

def drop_figure_labels(doc: ParsedDocument) -> ParsedDocument:
    """Drop blocks made only of very short lines, like the "1", "2", "a)" inside figures.

    Figure captions ("Fig. 45. Receptor olfactiv: ...") are long, so they stay.
    """
    pages = []
    for page in doc.pages:
        blocks = [block for block in page.blocks if not _is_label(block)]
        pages.append(page.model_copy(update={"blocks": blocks}))
    return doc.model_copy(update={"pages": pages})


def _is_label(block: Block) -> bool:
    return all(len(line.text.strip()) <= MAX_LABEL_CHARACTERS for line in block.lines)


# ---------- filter 3: words split across lines ----------

def join_hyphenated_lines(doc: ParsedDocument) -> ParsedDocument:
    """Inside a block, glue "chemorecep-" + "tori sunt" into "chemoreceptori sunt".

    Only when a line ends with a letter followed by "-" and the next line starts
    with a lowercase letter. Known trade-off: a real hyphen that falls at a line
    break ("într-" + "o") is joined too ("întro"); it is rare.
    The joined line keeps the first line's top and the second line's bottom.
    """
    pages = []
    for page in doc.pages:
        blocks = [block.model_copy(update={"lines": _join_lines(block.lines)}) for block in page.blocks]
        pages.append(page.model_copy(update={"blocks": blocks}))
    return doc.model_copy(update={"pages": pages})


def _join_lines(lines: list[Line]) -> list[Line]:
    joined: list[Line] = []
    for line in lines:
        previous = joined[-1] if joined else None
        if previous is not None and _ends_split_word(previous.text) and line.text[:1].islower():
            joined[-1] = Line(
                text=previous.text[:-1] + line.text,
                y_top=previous.y_top,
                y_bottom=line.y_bottom,
            )
        else:
            joined.append(line)
    return joined


def _ends_split_word(text: str) -> bool:
    return len(text) >= 2 and text.endswith("-") and text[-2].isalpha()
