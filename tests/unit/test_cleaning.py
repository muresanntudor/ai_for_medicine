from rag.domain.models import Block, Line, PageText, ParsedDocument, compute_page_offsets
from rag.indexing.cleaning import (
    clean,
    drop_figure_labels,
    join_hyphenated_lines,
    remove_headers_footers,
)

HEADER = "Funcțiile fundamentale ale organismului uman"


# ---------- helpers: build documents by hand ----------

def line(text: str, y: float) -> Line:
    return Line(text=text, y_top=y, y_bottom=y + 10)


def page(index: int, *blocks: Block) -> PageText:
    return PageText(page_index=index, page_label=str(index), extraction="text", blocks=list(blocks))


def paragraph(*lines: Line) -> Block:
    return Block(kind="paragraph", lines=list(lines))


def doc(*pages: PageText) -> ParsedDocument:
    pages = list(pages)
    return ParsedDocument(
        doc_id="id",
        title="Carte",
        file_name="carte.pdf",
        outline=[],
        pages=pages,
        page_offsets=compute_page_offsets(pages),
    )


def book_page(index: int, header: str = HEADER) -> PageText:
    """A page with a header at the top, content in the middle, page number at the bottom."""
    return page(
        index,
        paragraph(line(header, 20)),
        paragraph(line(f"Conținut real pe pagina {index}.", 100)),
        paragraph(line(str(index + 40), 700)),
    )


def texts(p: PageText) -> list[str]:
    return [ln.text for block in p.blocks for ln in block.lines]


# ---------- remove_headers_footers ----------

def test_repeated_header_and_page_numbers_are_removed():
    cleaned = remove_headers_footers(doc(*(book_page(i) for i in range(1, 5))))
    for i, p in enumerate(cleaned.pages, start=1):
        assert texts(p) == [f"Conținut real pe pagina {i}."]


def test_header_text_is_saved_as_section():
    cleaned = remove_headers_footers(doc(*(book_page(i) for i in range(1, 5))))
    assert all(p.section == HEADER for p in cleaned.pages)


def test_line_repeated_on_only_two_pages_stays():
    pages = [book_page(1, "Unic A"), book_page(2, "Unic B"), book_page(3, "Rar"), book_page(4, "Rar")]
    cleaned = remove_headers_footers(doc(*pages))
    assert texts(cleaned.pages[2])[0] == "Rar"
    assert cleaned.pages[2].section == ""


def test_short_documents_are_left_alone():
    original = doc(book_page(1), book_page(2))
    assert remove_headers_footers(original) == original


def test_only_top_and_bottom_lines_are_candidates():
    # The header text also appears in the middle of page 2: that occurrence is real content.
    pages = [book_page(i) for i in range(1, 5)]
    pages[1] = page(2, paragraph(line(HEADER, 20)), paragraph(line(HEADER, 300)), paragraph(line("42", 700)))
    cleaned = remove_headers_footers(doc(*pages))
    assert texts(cleaned.pages[1]) == [HEADER]


def test_original_document_is_not_changed():
    original = doc(*(book_page(i) for i in range(1, 5)))
    remove_headers_footers(original)
    assert texts(original.pages[0])[0] == HEADER


# ---------- drop_figure_labels ----------

def test_figure_labels_are_dropped_and_captions_stay():
    p = page(
        1,
        paragraph(line("1", 300)),
        paragraph(line("2", 320), line("a)", 330)),
        paragraph(line("Fig. 45. Receptor olfactiv: 1. fibre nervoase", 600)),
    )
    cleaned = drop_figure_labels(doc(p))
    assert texts(cleaned.pages[0]) == ["Fig. 45. Receptor olfactiv: 1. fibre nervoase"]


# ---------- join_hyphenated_lines ----------

def test_split_word_is_joined():
    p = page(1, paragraph(line("Receptorii sunt chemorecep-", 100), line("tori olfactivi.", 112)))
    cleaned = join_hyphenated_lines(doc(p))
    joined = cleaned.pages[0].blocks[0].lines
    assert [ln.text for ln in joined] == ["Receptorii sunt chemoreceptori olfactivi."]
    assert (joined[0].y_top, joined[0].y_bottom) == (100, 122)


def test_hyphen_before_uppercase_is_kept():
    p = page(1, paragraph(line("Sistem-", 100), line("Nervos central", 112)))
    assert texts(join_hyphenated_lines(doc(p)).pages[0]) == ["Sistem-", "Nervos central"]


def test_dash_after_space_is_kept():
    p = page(1, paragraph(line("Olfacția -", 100), line("simțul mirosului", 112)))
    assert texts(join_hyphenated_lines(doc(p)).pages[0]) == ["Olfacția -", "simțul mirosului"]


def test_word_split_three_times_is_joined():
    p = page(1, paragraph(line("hiper-", 100), line("vas-", 112), line("cularizat", 124)))
    assert texts(join_hyphenated_lines(doc(p)).pages[0]) == ["hipervascularizat"]


# ---------- clean: all together ----------

def test_clean_runs_all_filters_and_recomputes_offsets():
    pages = [book_page(i) for i in range(1, 5)]
    pages[0] = page(
        1,
        paragraph(line(HEADER, 20)),
        paragraph(line("Simțul mirosu-", 100), line("lui este slab.", 112)),
        paragraph(line("3", 300)),
        paragraph(line("41", 700)),
    )
    cleaned = clean(doc(*pages))
    assert texts(cleaned.pages[0]) == ["Simțul mirosului este slab."]
    assert cleaned.pages[0].section == HEADER
    assert cleaned.page_offsets == compute_page_offsets(cleaned.pages)
    assert cleaned.page_offsets != doc(*pages).page_offsets


def test_clean_keeps_every_page_even_if_empty():
    pages = [page(i, paragraph(line(HEADER, 20)), paragraph(line(str(i), 700))) for i in range(1, 5)]
    cleaned = clean(doc(*pages))
    assert [p.page_index for p in cleaned.pages] == [1, 2, 3, 4]
    assert all(p.blocks == [] for p in cleaned.pages)


def test_single_letter_header_is_removed_but_not_used_as_section():
    pages = [book_page(i, header="A") for i in range(1, 5)]
    cleaned = remove_headers_footers(doc(*pages))
    assert all(texts(p) == [f"Conținut real pe pagina {p.page_index}."] for p in cleaned.pages)
    assert all(p.section == "" for p in cleaned.pages)
