import pytest

from rag.domain.errors import CorruptPdf, ExtractionFailed


def test_extraction_failed_keeps_name_and_reason():
    err = ExtractionFailed("a.pdf", "OCR crashed on page 3")
    assert err.name == "a.pdf"
    assert err.reason == "OCR crashed on page 3"
    assert str(err) == "a.pdf: OCR crashed on page 3"


def test_corrupt_pdf_has_fixed_reason():
    err = CorruptPdf("a.pdf")
    assert err.name == "a.pdf"
    assert err.reason == "PDF could not be opened or read"


def test_corrupt_pdf_is_caught_as_extraction_failed():
    # The pipeline catches one type per PDF; subclasses must be caught too.
    with pytest.raises(ExtractionFailed):
        raise CorruptPdf("a.pdf")
