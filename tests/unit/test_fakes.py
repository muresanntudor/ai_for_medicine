import asyncio

import pytest

from rag.adapters.fakes.fake_parser import FakeParser
from rag.adapters.fakes.fake_source import FakeSource
from rag.domain.errors import CorruptPdf, ExtractionFailed
from rag.domain.models import ParsedDocument, SourceRef
from rag.ports.document_source import DocumentSource
from rag.ports.pdf_parser import PdfParser


# ---------- FakeSource ----------

def test_source_lists_every_file_sorted():
    source: DocumentSource = FakeSource({"b.pdf": b"%PDF b", "a.pdf": b"%PDF a"})
    refs = asyncio.run(source.list())
    assert [r.uri for r in refs] == ["a.pdf", "b.pdf"]
    assert all(r.allowed_groups == ["*"] for r in refs)


def test_source_uses_given_groups():
    source = FakeSource({"a.pdf": b"%PDF"}, allowed_groups=["cardiologie"])
    refs = asyncio.run(source.list())
    assert refs[0].allowed_groups == ["cardiologie"]


def test_source_empty_lists_nothing():
    assert asyncio.run(FakeSource({}).list()) == []


def test_source_reads_bytes():
    source = FakeSource({"a.pdf": b"%PDF a"})
    assert asyncio.run(source.read(SourceRef(uri="a.pdf"))) == b"%PDF a"


def test_source_missing_file_raises_extraction_failed():
    source = FakeSource({})
    with pytest.raises(ExtractionFailed) as err:
        asyncio.run(source.read(SourceRef(uri="gone.pdf")))
    assert err.value.reason == "file could not be read"


def test_source_copies_input_dict():
    files = {"a.pdf": b"%PDF"}
    source = FakeSource(files)
    files["b.pdf"] = b"%PDF"
    assert len(asyncio.run(source.list())) == 1


# ---------- FakeParser ----------

def test_parser_returns_valid_document():
    parser: PdfParser = FakeParser(pages=3)
    doc = parser.parse(b"%PDF data", "id-1", "guide.pdf")
    assert isinstance(doc, ParsedDocument)
    assert doc.doc_id == "id-1"
    assert doc.file_name == "guide.pdf"
    assert doc.title == "guide"
    assert [p.page_index for p in doc.pages] == [1, 2, 3]
    assert doc.page_offsets[0] == 0
    assert doc.page_offsets == sorted(doc.page_offsets)


def test_parser_offsets_match_text_lengths():
    doc = FakeParser(pages=2).parse(b"%PDF", "id", "a.pdf")
    first_page_text = doc.pages[0].blocks[0].lines[0].text
    assert doc.page_offsets[1] == len(first_page_text)


def test_parser_zero_pages_is_allowed():
    doc = FakeParser(pages=0).parse(b"%PDF", "id", "a.pdf")
    assert doc.pages == [] and doc.page_offsets == []


@pytest.mark.parametrize("data", [b"", b"not a pdf"])
def test_parser_bad_bytes_raise_corrupt_pdf(data):
    with pytest.raises(CorruptPdf):
        FakeParser().parse(data, "id", "bad.pdf")


def test_parser_fail_on_raises_extraction_failed():
    parser = FakeParser(fail_on={"broken.pdf"})
    with pytest.raises(ExtractionFailed) as err:
        parser.parse(b"%PDF", "id", "broken.pdf")
    assert not isinstance(err.value, CorruptPdf)


def test_parser_records_calls():
    parser = FakeParser()
    parser.parse(b"%PDF", "id", "a.pdf")
    with pytest.raises(CorruptPdf):
        parser.parse(b"", "id", "b.pdf")
    assert parser.calls == ["a.pdf", "b.pdf"]


def test_parser_runs_in_background_thread():
    # The pipeline will call parse via asyncio.to_thread.
    doc = asyncio.run(asyncio.to_thread(FakeParser().parse, b"%PDF", "id", "a.pdf"))
    assert doc.file_name == "a.pdf"


# ---------- Together: source -> parser, like the pipeline ----------

def test_source_and_parser_together():
    source = FakeSource({"a.pdf": b"%PDF a", "bad.pdf": b"junk"})
    parser = FakeParser()

    async def run():
        parsed, failed = [], []
        for ref in await source.list():
            data = await source.read(ref)
            try:
                parsed.append(await asyncio.to_thread(parser.parse, data, "id", ref.uri))
            except ExtractionFailed as e:
                failed.append(e.name)
        return parsed, failed

    parsed, failed = asyncio.run(run())
    assert [d.file_name for d in parsed] == ["a.pdf"]
    assert failed == ["bad.pdf"]
