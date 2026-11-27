"""
Turns a PDF's raw bytes into a ParsedDocument:
pages, blocks of text, each line with its position on the page, page labels, table of contents, page offsets.
The only place in the project that knows that PDFs exist.
Everything after it works with clean domain objects
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pymupdf
from rag.adapters.pdf.diacritics import fix_legacy_romanian, keep_text
from rag.adapters.pdf.page_strategy import choose_extraction
from rag.domain.models import PAGE_SEPARATOR, Block, Line, OutlineEntry, PageText, ParsedDocument
from rag.domain.errors import CorruptPdf, ExtractionFailed

# In PyMuPDF's "dict" output, a block of type 0 is text (1 would be an image).
TEXT_BLOCK = 0

# How much bigger than body text a block must be to count as a (sub)heading.
HEADING_RATIO = 1.3
SUBHEADING_RATIO = 1.1
# A bold block at body size is a subheading only if it is this short.
MAX_BOLD_SUBHEADING_LINES = 2

BlockKind = Literal["heading", "subheading", "paragraph"]


class PyMuPdfParser:
    """Reads PDFs with PyMuPDF and turns them into ParsedDocuments."""

    def __init__(self, fix_legacy_diacritics: bool = True):
        # Our PDFs use old Romanian fonts whose text needs repairing (see diacritics.py).
        # Pass False for PDFs whose text is already correct.
        self._fix_text: Callable[[str], str] = fix_legacy_romanian if fix_legacy_diacritics else keep_text

    def parse(self, data: bytes, doc_id: str, file_name: str) -> ParsedDocument:
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # PyMuPDF raises several error types for broken files
            raise CorruptPdf(file_name) from exc

        with pdf:
            try:
                #  find the size of normal body text, used later to spot headings.
                body_size = _body_font_size(pdf)

                # Build every page, numbered from 1 (page_index must be positive).
                pages = []
                for page_index, page in enumerate(pdf, start = 1):
                    pages.append(_build_page(page, page_index, body_size, self._fix_text))

                return ParsedDocument(
                    doc_id=doc_id,
                    title=_title(pdf, file_name, self._fix_text),
                    file_name=file_name,
                    outline=_outline(pdf, self._fix_text),
                    pages=pages,
                    page_offsets=_page_offsets(pages),
                )
            except ExtractionFailed:
                # Our own errors (CorruptPdf included) already have a clear reason.
                raise
            except Exception as exc:
                # Safety net: anything else becomes one error type the pipeline can catch,
                # so one bad PDF never stops the others.
                raise ExtractionFailed(file_name, "unexpected error while parsing") from exc


def _body_font_size(pdf: pymupdf.Document) -> float | None:
    """
    Return the font size used for most of the text in the PDF.
    For differencing between headings and subheadngs
    """
    votes: Counter[float] = Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != TEXT_BLOCK:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    characters = len(span["text"].strip())
                    if characters:
                        votes[round(span["size"], 1)] += characters
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def _block_kind(block: dict, body_size: float | None) -> BlockKind:
    """Decide if one text block (from page.get_text("dict")) is a heading,
    a subheading or a normal paragraph, from its font size and boldness.

    1. at least 1.3x body size           -> heading
    2. at least 1.1x body size           -> subheading
    3. body size, ALL bold, 1-2 lines    -> subheading
    4. anything else                     -> paragraph
    """
    spans = [
        span
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    if not spans or body_size is None:
        return "paragraph"

    biggest = max(round(span["size"], 1) for span in spans)
    if biggest >= body_size * HEADING_RATIO:
        return "heading"
    if biggest >= body_size * SUBHEADING_RATIO:
        return "subheading"

    lines_with_text = [
        line for line in block["lines"]
        if any(span["text"].strip() for span in line["spans"])
    ]
    if len(lines_with_text) <= MAX_BOLD_SUBHEADING_LINES and all(_is_bold(s) for s in spans):
        return "subheading"
    return "paragraph"


def _is_bold(span: dict) -> bool:
    """Bold is either flagged by the PDF or only visible in the font's name."""
    return bool(span["flags"] & pymupdf.TEXT_FONT_BOLD) or "bold" in span["font"].lower()


def _build_page(
    page: pymupdf.Page,
    page_index: int,
    body_size: float | None,
    fix_text: Callable[[str], str] = keep_text,
) -> PageText:
    """Turn one PyMuPDF page into a PageText.

    `page_index` counts from 1. Images are skipped for now, empty lines and
    empty blocks are dropped, and each block gets its kind from _block_kind.
    """
    blocks: list[Block] = []
    for raw_block in page.get_text("dict")["blocks"]:
        if raw_block["type"] != TEXT_BLOCK:
            continue

        lines: list[Line] = []
        for raw_line in raw_block["lines"]:
            text = fix_text("".join(span["text"] for span in raw_line["spans"])).strip()
            if not text:
                continue
            # bbox is (x0, y_top, x1, y_bottom) in points from the top of the page.
            lines.append(Line(text=text, y_top=raw_line["bbox"][1], y_bottom=raw_line["bbox"][3]))

        if lines:
            blocks.append(Block(kind=_block_kind(raw_block, body_size), lines=lines))

    page_text = "\n".join(line.text for block in blocks for line in block.lines)

    return PageText(
        page_index=page_index,
        # The number printed in the book, if the PDF defines one; else the plain index.
        page_label=page.get_label() or str(page_index),
        extraction=choose_extraction(page_text),
        blocks=blocks,
    )

def _title(pdf: pymupdf.Document, file_name: str, fix_text: Callable[[str], str] = keep_text) -> str:
    """The title stored in the PDF, or the file name without .pdf if there is none."""
    metadata = pdf.metadata or {}
    title = fix_text(metadata.get("title") or "").strip()
    return title or Path(file_name).stem


def _outline(pdf: pymupdf.Document, fix_text: Callable[[str], str] = keep_text) -> list[OutlineEntry]:
    """The PDF's built-in table of contents, or an empty list if it has none."""
    entries: list[OutlineEntry] = []
    for level, title, page in pdf.get_toc():
        title = fix_text(title).strip()
        if not title or page < 1:
            continue
        entries.append(OutlineEntry(level=level, title=title, page_index=page))
    return entries


def _page_offsets(pages: list[PageText]) -> list[int]:
    """Where each page starts in the whole document's text.

    The document text is every page's plain_text() joined with PAGE_SEPARATOR,
    so each next page starts after the previous page's text plus the separator.
    """
    offsets: list[int] = []
    position = 0
    for page in pages:
        offsets.append(position)
        position += len(page.plain_text()) + len(PAGE_SEPARATOR)
    return offsets
