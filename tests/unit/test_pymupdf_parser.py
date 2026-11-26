import pymupdf
import pytest

from rag.adapters.pdf.pymupdf_parser import (
    PyMuPdfParser,
    _block_kind,
    _body_font_size,
    _build_page,
    _outline,
    _page_offsets,
    _title,
)
from rag.domain.errors import CorruptPdf, ExtractionFailed
from rag.domain.models import PAGE_SEPARATOR, Block, Line, PageText, ParsedDocument
from rag.ports.pdf_parser import PdfParser


def make_pdf(pages: list[list[tuple[str, float]]]) -> bytes:
    """Build a small PDF in memory: one list of (text, font size) lines per page."""
    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        y = 72
        for text, size in lines:
            page.insert_text((72, y), text, fontsize=size)
            y += size * 2
    data = pdf.tobytes()
    pdf.close()
    return data


def test_junk_bytes_raise_corrupt_pdf():
    with pytest.raises(CorruptPdf):
        PyMuPdfParser().parse(b"junk", "id", "bad.pdf")


def test_body_size_is_the_most_used_size():
    data = make_pdf([
        [("Cardiologie", 20), ("Inima pompeaza sangele in tot corpul.", 11),
         ("Ritmul normal este intre 60 si 100 de batai pe minut.", 11)],
        [("Capitolul 2", 20), ("Text normal pe a doua pagina.", 11)],
    ])
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert _body_font_size(pdf) == 11


def test_body_size_counts_characters_not_lines():
    # Three short big lines vs one long small line: the long one wins.
    data = make_pdf([[("A", 20), ("B", 20), ("C", 20), ("x" * 80, 9)]])
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert _body_font_size(pdf) == 9


def test_body_size_is_none_without_text():
    data = make_pdf([[]])
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert _body_font_size(pdf) is None


# ---------- _block_kind: font size + bold rule ----------

def block(*lines: list[tuple[str, float, bool]]) -> dict:
    """Fake one block the way page.get_text("dict") returns it.

    Each line is a list of spans: (text, size, bold).
    """
    return {
        "type": 0,
        "lines": [
            {"spans": [
                {"text": text, "size": size, "font": "Helvetica",
                 "flags": pymupdf.TEXT_FONT_BOLD if bold else 0}
                for text, size, bold in spans
            ]}
            for spans in lines
        ],
    }


def test_much_bigger_font_is_heading():
    assert _block_kind(block([("Cardiologie", 15, False)]), 11) == "heading"


def test_slightly_bigger_font_is_subheading():
    assert _block_kind(block([("Aritmii", 12.5, False)]), 11) == "subheading"


def test_body_size_bold_short_block_is_subheading():
    assert _block_kind(block([("Fibrilatia atriala", 11, True)]), 11) == "subheading"


def test_bold_word_inside_paragraph_stays_paragraph():
    b = block([("Atentie: ", 11, True), ("doza maxima este 2 g pe zi.", 11, False)])
    assert _block_kind(b, 11) == "paragraph"


def test_long_bold_block_stays_paragraph():
    b = block([("Avertisment lung", 11, True)], [("pe mai", 11, True)], [("multe randuri", 11, True)])
    assert _block_kind(b, 11) == "paragraph"


def test_bold_only_in_font_name_counts():
    b = block([("Diagnostic", 11, False)])
    b["lines"][0]["spans"][0]["font"] = "Arial-BoldMT"
    assert _block_kind(b, 11) == "subheading"


def test_normal_text_is_paragraph():
    assert _block_kind(block([("Text normal.", 11, False)]), 11) == "paragraph"


def test_no_body_size_means_paragraph():
    assert _block_kind(block([("Ceva", 20, False)]), None) == "paragraph"


def test_real_pdf_bold_line_is_detected_as_subheading():
    # Build a real PDF with Helvetica Bold ("hebo") and run it through PyMuPDF.
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Tratament", fontsize=11, fontname="hebo")
    page.insert_text((72, 140), "Pacientul primeste tratament zilnic. " * 3, fontsize=11)
    with pymupdf.open(stream=pdf.tobytes(), filetype="pdf") as reopened:
        body = _body_font_size(reopened)
        blocks = [b for b in reopened[0].get_text("dict")["blocks"] if b["type"] == 0]
        kinds = [_block_kind(b, body) for b in blocks]
    assert kinds == ["subheading", "paragraph"]


# ---------- _build_page ----------

BODY = "Pacientul prezinta dureri toracice si dispnee de efort de doua saptamani."


def build_pages(data: bytes):
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        body = _body_font_size(pdf)
        return [_build_page(page, i, body) for i, page in enumerate(pdf, start=1)]


def test_build_page_numbers_and_labels():
    pages = build_pages(make_pdf([[("Cardiologie", 20), (BODY, 11)], [(BODY, 11)]]))
    assert [p.page_index for p in pages] == [1, 2]
    assert [p.page_label for p in pages] == ["1", "2"]


def test_build_page_blocks_lines_and_kinds():
    page = build_pages(make_pdf([[("Cardiologie", 20), (BODY, 11)]]))[0]
    assert [b.kind for b in page.blocks] == ["heading", "paragraph"]
    title = page.blocks[0].lines[0]
    assert title.text == "Cardiologie"
    assert 0 < title.y_top < title.y_bottom
    assert page.blocks[1].lines[0].text == BODY
    assert page.extraction == "text"


def test_build_page_uses_label_printed_in_pdf():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), BODY, fontsize=11)
    pdf.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}])
    pages = build_pages(pdf.tobytes())
    assert pages[0].page_label == "i"


def test_build_page_empty_page_needs_ocr():
    page = build_pages(make_pdf([[]]))[0]
    assert page.blocks == []
    assert page.extraction == "ocr"


# ---------- _title and _outline ----------

def reopen(pdf: pymupdf.Document) -> pymupdf.Document:
    """Save and reopen, so we test what a real PDF file contains."""
    return pymupdf.open(stream=pdf.tobytes(), filetype="pdf")


def test_title_from_metadata():
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.set_metadata({"title": "  Cardiologie  "})
    with reopen(pdf) as saved:
        assert _title(saved, "carte.pdf") == "Cardiologie"


@pytest.mark.parametrize("file_name, expected", [("carte.pdf", "carte"), ("cardio/ecg.pdf", "ecg")])
def test_title_falls_back_to_file_name(file_name: str, expected: str):
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.set_metadata({"title": "   "})
    with reopen(pdf) as saved:
        assert _title(saved, file_name) == expected


def test_outline_from_table_of_contents():
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.new_page()
    pdf.set_toc([[1, "Capitol 1", 1], [2, "Sectiune", 2]])
    with reopen(pdf) as saved:
        outline = _outline(saved)
    assert [(e.level, e.title, e.page_index) for e in outline] == [
        (1, "Capitol 1", 1),
        (2, "Sectiune", 2),
    ]


def test_outline_empty_without_table_of_contents():
    with pymupdf.open(stream=make_pdf([[("Text", 11)]]), filetype="pdf") as pdf:
        assert _outline(pdf) == []


def test_outline_skips_broken_rows():
    toc = [[1, "Bun", 1], [1, "   ", 1], [1, "Pagina rupta", -1]]

    class FakePdf:
        def get_toc(self):
            return toc

    assert [e.title for e in _outline(FakePdf())] == ["Bun"]


# ---------- _page_offsets ----------

def text_page(index: int, text: str) -> PageText:
    blocks = [Block(kind="paragraph", lines=[Line(text=text, y_top=0, y_bottom=1)])] if text else []
    return PageText(page_index=index, page_label=str(index), extraction="text", blocks=blocks)


def test_offsets_no_pages():
    assert _page_offsets([]) == []


def test_offsets_one_page():
    assert _page_offsets([text_page(1, "abc")]) == [0]


def test_offsets_add_length_plus_separator():
    pages = [text_page(1, "a" * 10), text_page(2, "b" * 5), text_page(3, "c")]
    assert _page_offsets(pages) == [0, 12, 19]


def test_offsets_with_empty_page():
    pages = [text_page(1, "abc"), text_page(2, ""), text_page(3, "xyz")]
    assert _page_offsets(pages) == [0, 5, 7]


def test_offsets_point_at_each_page_in_the_full_text():
    pdf_bytes = make_pdf([
        [("Cardiologie", 20), (BODY, 11)],
        [(BODY, 11), ("Al doilea paragraf.", 11)],
        [("Capitol final", 20)],
    ])
    pages = build_pages(pdf_bytes)
    full_text = PAGE_SEPARATOR.join(page.plain_text() for page in pages)
    for page, offset in zip(pages, _page_offsets(pages)):
        assert full_text[offset:].startswith(page.plain_text())


# ---------- parse: the whole thing end to end ----------

def test_parse_returns_full_parsed_document():
    pdf = pymupdf.open()
    for title_line in ("Capitol 1", "Capitol 2"):
        page = pdf.new_page()
        page.insert_text((72, 72), title_line, fontsize=20)
        page.insert_text((72, 140), BODY, fontsize=11)
    pdf.set_metadata({"title": "Cardiologie"})
    pdf.set_toc([[1, "Capitol 1", 1], [1, "Capitol 2", 2]])

    parser: PdfParser = PyMuPdfParser()
    doc = parser.parse(pdf.tobytes(), "doc-1", "cardio/carte.pdf")

    assert isinstance(doc, ParsedDocument)
    assert doc.doc_id == "doc-1"
    assert doc.file_name == "cardio/carte.pdf"
    assert doc.title == "Cardiologie"
    assert [e.title for e in doc.outline] == ["Capitol 1", "Capitol 2"]
    assert [p.page_index for p in doc.pages] == [1, 2]
    assert [b.kind for b in doc.pages[0].blocks] == ["heading", "paragraph"]
    assert doc.page_offsets == _page_offsets(doc.pages)


def test_parse_unexpected_error_becomes_extraction_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("something odd inside PyMuPDF")

    monkeypatch.setattr("rag.adapters.pdf.pymupdf_parser._build_page", boom)
    with pytest.raises(ExtractionFailed) as err:
        PyMuPdfParser().parse(make_pdf([[(BODY, 11)]]), "id", "odd.pdf")
    assert err.value.reason == "unexpected error while parsing"
    assert isinstance(err.value.__cause__, RuntimeError)


def test_parse_lets_own_errors_through_unchanged(monkeypatch):
    def corrupt(*args, **kwargs):
        raise CorruptPdf("x.pdf")

    monkeypatch.setattr("rag.adapters.pdf.pymupdf_parser._build_page", corrupt)
    with pytest.raises(CorruptPdf):
        PyMuPdfParser().parse(make_pdf([[(BODY, 11)]]), "id", "x.pdf")


# ---------- legacy Romanian diacritics ----------

def legacy_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Func]iile fundamentale", fontsize=20)
    page.insert_text((72, 140), "Sim]ul mirosului este slab dezvoltat la om [i la alte specii.", fontsize=11)
    pdf.set_metadata({"title": "Biologie pentru clasa a XI-a [i a XII-a"})
    pdf.set_toc([[1, "Func]iile fundamentale", 1]])
    return pdf.tobytes()


def test_parser_fixes_diacritics_by_default():
    doc = PyMuPdfParser().parse(legacy_pdf(), "id", "carte.pdf")
    texts = [line.text for block in doc.pages[0].blocks for line in block.lines]
    assert texts == ["Funcțiile fundamentale", "Simțul mirosului este slab dezvoltat la om și la alte specii."]
    assert doc.title == "Biologie pentru clasa a XI-a și a XII-a"
    assert doc.outline[0].title == "Funcțiile fundamentale"


def test_parser_can_leave_text_untouched():
    doc = PyMuPdfParser(fix_legacy_diacritics=False).parse(legacy_pdf(), "id", "carte.pdf")
    assert doc.pages[0].blocks[0].lines[0].text == "Func]iile fundamentale"
