"""
Repairs Romanian letters that old PDFs store under the wrong characters.

Many Romanian books typeset before ~2010 used "legacy" fonts (names ending in
"Rom", e.g. Korinthia-Rom). They draw ă, ș, ț on screen, but store other
characters in the text layer, e.g. "Funcțiile" is stored as "Func]iile" or
"Funcþiile". Extracted text would never match a normally typed question,
so we translate those characters back before anything else sees the text.
"""

# Two legacy encodings, both found in data/A367.pdf.
_LEGACY_ROMANIAN = str.maketrans({
    # ISO-8859-1 look-alikes
    "ã": "ă", "Ã": "Ă",
    "þ": "ț", "Þ": "Ț",
    "º": "ș", "ª": "Ș",
    # ASCII-symbol slots
    "\\": "ă", "|": "Ă",
    "]": "ț", "}": "Ț",
    "[": "ș", "{": "Ș",
    "`": "î", "~": "Î",
    # Old cedilla letters -> correct comma-below letters (so "şi" == "și" in search)
    "ş": "ș", "Ş": "Ș",
    "ţ": "ț", "Ţ": "Ț",
})


def fix_legacy_romanian(text: str) -> str:
    """Return `text` with legacy-font characters turned into real Romanian letters.

    Only use it on PDFs that need it: it also turns real brackets [ ] { } and
    backslashes into letters.
    """
    return text.translate(_LEGACY_ROMANIAN)


def keep_text(text: str) -> str:
    """The 'do nothing' fixer, for PDFs whose text is already correct."""
    return text
