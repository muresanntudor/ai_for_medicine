"""where the pipeline gets PDFs from, without knowing if it's a local folder, S3, etc."""

from typing import Protocol
from rag.domain.models import SourceRef


class DocumentSource(Protocol):
    """Any place PDFs can be listed and read from.

    Adapters (e.g. a local folder) implement these two methods; the pipeline
    only ever talks to this contract, never to the concrete source.
    """

    def list(self) -> list[SourceRef]:
        """Return a SourceRef for every PDF this source can see.

        Returns an empty list when there are no PDFs. Must not read file contents.
        """
        ...

    def read(self, ref: SourceRef) -> bytes:
        """Return the raw bytes of the PDF that `ref` points to.

        Raises ExtractionFailed (reason "file could not be read") when the file
        is missing or cannot be opened, so the pipeline catches one error type
        per PDF and moves on to the next.
        """
        ...
