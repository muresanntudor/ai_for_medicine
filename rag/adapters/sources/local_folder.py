"""DocumentSource that reads PDFs from a folder on the local disk (e.g. data/pdfs)."""

# Postpones type hints, because the method named `list` hides the built-in
# `list` inside the class body (so `list[str]` would break).
from __future__ import annotations

from pathlib import Path

from rag.domain.errors import ExtractionFailed
from rag.domain.models import SourceRef


class LocalFolderSource:
    """
    Librarian: Only finds the files and fetches them
    1. Which PDFs we have here? Gives back a list with each name inside it.
    2. Give me this one: and sends PDF's raw bytes (contents)
    """

    def __init__(self, path: str | Path, allowed_groups: list[str] | None = None):
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            # A missing folder is a setup mistake, not a bad PDF.
            raise ValueError(f"PDF folder does not exist: {root}")
        self.path = root
        self._allowed_groups = allowed_groups or ["*"]

    def list(self) -> list[SourceRef]:
        # What PDFs are there? Looks inside the folder, finds every PDF and makes one SourceRef/ PDF
        return [SourceRef(uri=uri, allowed_groups=self._allowed_groups) for uri in self._scan()]

    def read(self, ref: SourceRef) -> bytes:
        #Gives PDF's byte
        file = self._resolve(ref.uri)
        try:
            return file.read_bytes()
        except OSError as exc:  # missing, no permission, is a folder, ...
            raise ExtractionFailed(ref.uri, "file could not be read") from exc

    def _scan(self) -> list[str]:
        uris = []
        for file in self.path.rglob("*"):
            relative = file.relative_to(self.path)
            # Skip hidden files and folders, e.g. macOS "._report.pdf" junk.
            if any(part.startswith(".") for part in relative.parts):
                continue
            if file.is_file() and file.suffix.lower() == ".pdf":
                uris.append(relative.as_posix())
        return sorted(uris)

    def _resolve(self, uri: str) -> Path:
        """Turn a uri back into a full path, refusing anything outside the folder."""
        file = (self.path / uri).resolve()
        if not file.is_relative_to(self.path):
            raise ExtractionFailed(uri, "file could not be read")
        return file
