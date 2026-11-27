from pathlib import Path

import pytest

from rag.adapters.sources.local_folder import LocalFolderSource
from rag.domain.errors import ExtractionFailed
from rag.domain.models import SourceRef
from rag.ports.document_source import DocumentSource


@pytest.fixture
def pdf_folder(tmp_path: Path) -> Path:
    (tmp_path / "b.pdf").write_bytes(b"%PDF b")
    (tmp_path / "cardio").mkdir()
    (tmp_path / "cardio" / "A.PDF").write_bytes(b"%PDF a")
    (tmp_path / "notes.txt").write_bytes(b"not a pdf")
    (tmp_path / "._b.pdf").write_bytes(b"mac junk")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.pdf").write_bytes(b"%PDF c")
    (tmp_path / "folder.pdf").mkdir()  # a folder named like a PDF
    return tmp_path


def test_missing_folder_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        LocalFolderSource(tmp_path / "nope")


def test_lists_only_real_pdfs_sorted_with_relative_uris(pdf_folder: Path):
    source: DocumentSource = LocalFolderSource(pdf_folder)
    refs = source.list()
    assert [r.uri for r in refs] == ["b.pdf", "cardio/A.PDF"]
    assert all(r.allowed_groups == ["*"] for r in refs)


def test_uses_given_groups(pdf_folder: Path):
    refs = LocalFolderSource(pdf_folder, allowed_groups=["cardiologie"]).list()
    assert all(r.allowed_groups == ["cardiologie"] for r in refs)


def test_empty_folder_lists_nothing(tmp_path: Path):
    assert LocalFolderSource(tmp_path).list() == []


def test_reads_bytes_of_listed_files(pdf_folder: Path):
    source = LocalFolderSource(pdf_folder)
    refs = source.list()
    data = [source.read(r) for r in refs]
    assert data == [b"%PDF b", b"%PDF a"]


def test_accepts_str_path(pdf_folder: Path):
    source = LocalFolderSource(str(pdf_folder))
    assert source.read(SourceRef(uri="b.pdf")) == b"%PDF b"


@pytest.mark.parametrize("uri", ["gone.pdf", "folder.pdf", "../outside.pdf"])
def test_unreadable_uri_raises_extraction_failed(pdf_folder: Path, uri: str):
    (pdf_folder.parent / "outside.pdf").write_bytes(b"%PDF secret")
    source = LocalFolderSource(pdf_folder)
    with pytest.raises(ExtractionFailed) as err:
        source.read(SourceRef(uri=uri))
    assert err.value.reason == "file could not be read"
