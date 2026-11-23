"""Turns the raw bytes of one PDF into a ParsedDocument, without the pipeline
knowing which library (PyMuPDF, OCR, ...) does the work."""

from typing import Protocol
from rag.domain.models import ParsedDocument


class PdfParser(Protocol):
    """Anything that can read a PDF's bytes and return its structured content.

    parse() is a normal (sync) method on purpose: parsing is heavy CPU work,
    so the pipeline runs it in a background thread with asyncio.to_thread.
    """

    def parse(self, data: bytes, doc_id: str, file_name: str) -> ParsedDocument:
        """Parse one PDF and return its full ParsedDocument.

        `doc_id` comes from ids.doc_id(uri) and `file_name` is shown in citations;
        both are copied onto the returned ParsedDocument.

        Raises CorruptPdf when the bytes cannot be opened as a PDF, and
        ExtractionFailed for any other problem. Never returns a half-parsed
        document: it either succeeds completely or raises.
        """
        ...
