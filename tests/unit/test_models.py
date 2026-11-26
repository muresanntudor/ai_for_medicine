import pytest
from pydantic import ValidationError

from rag.domain.models import (
    PAGE_SEPARATOR,
    Block,
    Line,
    OutlineEntry,
    PageText,
    ParsedDocument,
    SourceRef,
)


def make_page(index: int, extraction: str = "text") -> PageText:
    line = Line(text=f"Text on page {index}", y_top=100, y_bottom=112)
    return PageText(
        page_index=index,
        page_label=str(index),
        extraction=extraction,
        blocks=[Block(kind="paragraph", lines=[line])],
    )


def make_document(**overrides) -> ParsedDocument:
    fields = dict(
        doc_id="346e9b52-9544-5e18-82a1-bcb35d80a7e3",
        title="Parser",
        file_name="parser.pdf",
        outline=[OutlineEntry(level=1, title="Intro", page_index=1)],
        pages=[make_page(1), make_page(2)],
        page_offsets=[0, 1200],
    )
    fields.update(overrides)
    return ParsedDocument(**fields)


# SourceRef

def test_source_ref_defaults_to_everyone():
    assert SourceRef(uri="/data/parser.pdf").allowed_groups == ["*"]


def test_source_ref_defaults_are_not_shared():
    a = SourceRef(uri="/a.pdf")
    b = SourceRef(uri="/b.pdf")
    assert a.allowed_groups is not b.allowed_groups


# Line

def test_line_accepts_decimal_positions():
    # PyMuPDF returns positions like 72.5, so Line must accept them.
    line = Line(text="Header", y_top=72.5, y_bottom=84.3)
    assert line.y_top == 72.5


# Block / PageText

def test_block_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        Block(kind="image", lines=[])


def test_page_rejects_unknown_extraction():
    with pytest.raises(ValidationError):
        make_page(1, extraction="scan")


def test_page_index_must_start_at_one():
    with pytest.raises(ValidationError):
        make_page(0)


def test_outline_level_must_be_positive():
    with pytest.raises(ValidationError):
        OutlineEntry(level=0, title="Intro", page_index=1)


# ParsedDocument

def test_valid_document_is_accepted():
    doc = make_document()
    assert len(doc.pages) == 2
    assert doc.page_offsets == [0, 1200]


def test_single_page_document_is_accepted():
    doc = make_document(pages=[make_page(1)], page_offsets=[0])
    assert len(doc.pages) == 1


def test_offsets_must_match_page_count():
    with pytest.raises(ValidationError, match="one entry per page"):
        make_document(page_offsets=[0])


def test_offsets_must_start_at_zero():
    with pytest.raises(ValidationError, match="start at 0"):
        make_document(page_offsets=[5, 1200])


def test_offsets_must_not_decrease():
    with pytest.raises(ValidationError, match="never decrease"):
        make_document(pages=[make_page(1), make_page(2), make_page(3)], page_offsets=[0, 1200, 900])


def test_equal_offsets_are_allowed_for_empty_pages():
    doc = make_document(pages=[make_page(1), make_page(2), make_page(3)], page_offsets=[0, 1200, 1200])
    assert doc.page_offsets[1] == doc.page_offsets[2]


# frozen and extra="forbid" (same config on every model)

def test_document_is_frozen():
    doc = make_document()
    with pytest.raises(ValidationError):
        doc.title = "Changed"


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        SourceRef(uri="/a.pdf", allowed_group=["doctors"])


# ---------- PageText.plain_text ----------

def test_plain_text_joins_lines_and_blocks():
    page = PageText(
        page_index=1,
        page_label="1",
        extraction="text",
        blocks=[
            Block(kind="heading", lines=[Line(text="a", y_top=0, y_bottom=1), Line(text="b", y_top=1, y_bottom=2)]),
            Block(kind="paragraph", lines=[Line(text="c", y_top=2, y_bottom=3)]),
        ],
    )
    assert page.plain_text() == "a\nb\n\nc"


def test_plain_text_of_empty_page_is_empty():
    assert PageText(page_index=1, page_label="1", extraction="ocr", blocks=[]).plain_text() == ""


def test_page_separator_is_a_blank_line():
    assert PAGE_SEPARATOR == "\n\n"
