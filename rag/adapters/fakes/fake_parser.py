"""PdfParser for tests: builds a small ParsedDocument without opening a real PDF."""

from rag.domain.errors import CorruptPdf, ExtractionFailed
from rag.domain.models import Block, Line, PageText, ParsedDocument


class FakeParser:
    """Implements the PdfParser port.

    - bytes that don't start with "%PDF" -> CorruptPdf
    - file names listed in `fail_on` -> ExtractionFailed
    - otherwise a ParsedDocument with `pages` pages, one paragraph each
    Every call is recorded in `calls` (file names) so tests can check what was parsed.
    """

    def __init__(self, pages: int = 2, fail_on: set[str] | None = None):
        self._pages = pages
        self._fail_on = fail_on or set()
        self.calls: list[str] = []

    def parse(self, data: bytes, doc_id: str, file_name: str) -> ParsedDocument:
        self.calls.append(file_name)
        if not data.startswith(b"%PDF"):
            raise CorruptPdf(file_name)
        if file_name in self._fail_on:
            raise ExtractionFailed(file_name, "fake failure")

        pages: list[PageText] = []
        page_offsets: list[int] = []
        offset = 0
        for number in range(1, self._pages + 1):
            text = f"Text of page {number} in {file_name}."
            line = Line(text=text, y_top=100.0, y_bottom=112.0)
            pages.append(
                PageText(
                    page_index=number,
                    page_label=str(number),
                    extraction="text",
                    blocks=[Block(kind="paragraph", lines=[line])],
                )
            )
            page_offsets.append(offset)
            offset += len(text)

        return ParsedDocument(
            doc_id=doc_id,
            title=file_name.removesuffix(".pdf"),
            file_name=file_name,
            outline=[],
            pages=pages,
            page_offsets=page_offsets,
        )
