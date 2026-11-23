"""
Computes the stable identities of a PDF, and
they let the indexer decide whether to skip, update or re-index a file.
"""
import uuid
import hashlib
import json

# Fixed seed for all doc_ids
DOC_NAMESPACE = uuid.UUID("73e97655-a101-4156-af38-625a86a3b336")

def _normalise_doc(uri: str) -> str:
    # "  /Data/Parser.pdf/ "  ->  "/data/parser.pdf"
    trimed = uri.strip().rstrip("/")
    normalised_uri= trimed.lower()
    return normalised_uri

def doc_id(uri: str)-> str:
    # "  /Data/Parser.pdf/ "  ->  "346e9b52-9544-5e18-82a1-bcb35d80a7e3"
    normalised_uri = _normalise_doc(uri)
    return str(uuid.uuid5(DOC_NAMESPACE, normalised_uri ))

def content_hash(data: bytes) -> str:
    # b"hello"  ->  "2cf24dba…938b9824"
    return hashlib.sha256(data).hexdigest()

def index_signature(parser_version:str, chunker_config:dict, embedding_fingerprint:str )-> str:
    """
    ("1.0", {"target": 350, "max": 512, "overlap": 50}, "openai/text-embedding-3-small@1:1536:true")  ->  "caca82d0…2769765"
    ("1.0", {"overlap": 50, "max": 512, "target": 350}, same model)
    """
    payload = json.dumps(
        [parser_version, chunker_config, embedding_fingerprint],
        sort_keys = True,
        separators=(",",":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

