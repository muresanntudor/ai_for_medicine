"""In-memory DocumentSource for tests: PDFs are kept in a dict instead of on disk."""

from rag.domain.errors import ExtractionFailed
from rag.domain.models import SourceRef


class FakeSource:
    """Implements the DocumentSource port with a dict of uri -> bytes."""

    def __init__(self, files: dict[str, bytes], allowed_groups: list[str] | None = None):
        self._files = dict(files)
        self._allowed_groups = allowed_groups or ["*"]

    async def list(self) -> list[SourceRef]:
        # Sorted so tests get the same order every run.
        return [
            SourceRef(uri=uri, allowed_groups=self._allowed_groups)
            for uri in sorted(self._files)
        ]

    async def read(self, ref: SourceRef) -> bytes:
        if ref.uri not in self._files:
            raise ExtractionFailed(ref.uri, "file could not be read")
        return self._files[ref.uri]
