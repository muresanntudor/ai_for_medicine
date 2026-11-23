import uuid

from rag.domain.ids import content_hash, doc_id, index_signature

SETTINGS = {"target": 350, "max": 512, "overlap": 50}
FINGERPRINT = "openai/text-embedding-3-small@1:1536:true"


# doc_id

def test_doc_id_same_path_same_id():
    assert doc_id("/data/parser.pdf") == doc_id("/data/parser.pdf")


def test_doc_id_ignores_case_spaces_and_trailing_slash():
    assert doc_id("  /Data/Parser.PDF/ ") == doc_id("/data/parser.pdf")


def test_doc_id_different_path_different_id():
    assert doc_id("/data/parser.pdf") != doc_id("/data/other.pdf")


def test_doc_id_is_a_valid_uuid_string():
    value = doc_id("/data/parser.pdf")
    assert isinstance(value, str)
    assert str(uuid.UUID(value)) == value


def test_doc_id_is_stable_across_runs():
    # Value computed once with DOC_NAMESPACE; if this fails, the namespace
    # or the normalisation changed and every stored doc_id would be orphaned.
    assert doc_id("/data/parser.pdf") == "346e9b52-9544-5e18-82a1-bcb35d80a7e3"


# content_hash

def test_content_hash_same_bytes_same_hash():
    assert content_hash(b"hello") == content_hash(b"hello")


def test_content_hash_one_byte_changed_different_hash():
    assert content_hash(b"hello") != content_hash(b"hellp")


def test_content_hash_is_sha256_hex():
    value = content_hash(b"hello")
    assert len(value) == 64
    assert value == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


# index_signature

def test_index_signature_ignores_key_order():
    reordered = {"overlap": 50, "max": 512, "target": 350}
    assert index_signature("1.0", SETTINGS, FINGERPRINT) == index_signature("1.0", reordered, FINGERPRINT)


def test_index_signature_changes_when_a_setting_changes():
    changed = {**SETTINGS, "target": 400}
    assert index_signature("1.0", SETTINGS, FINGERPRINT) != index_signature("1.0", changed, FINGERPRINT)


def test_index_signature_changes_when_parser_version_changes():
    assert index_signature("1.0", SETTINGS, FINGERPRINT) != index_signature("1.1", SETTINGS, FINGERPRINT)


def test_index_signature_changes_when_embedding_model_changes():
    assert index_signature("1.0", SETTINGS, FINGERPRINT) != index_signature("1.0", SETTINGS, "other/model@1:768:true")
