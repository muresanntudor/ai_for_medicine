import pytest

from rag.adapters.pdf.diacritics import fix_legacy_romanian, keep_text


@pytest.mark.parametrize("broken, fixed", [
    # ASCII-symbol encoding
    ("Func]iile", "Funcțiile"),
    ("[i", "și"),
    ("senzitiv\\", "senzitivă"),
    ("TEME {I APLICA}II", "TEME ȘI APLICAȚII"),
    ("ALC|TUIREA", "ALCĂTUIREA"),
    ("`n corp", "în corp"),
    ("~ncerca]i", "Încercați"),
    # ISO-8859-1 encoding
    ("Simþul", "Simțul"),
    ("Bucureºti", "București"),
    ("alcãtuiesc", "alcătuiesc"),
    ("EDUCAÞIEI ªI CERCETÃRII", "EDUCAȚIEI ȘI CERCETĂRII"),
    # old cedilla letters
    ("şi ţesut", "și țesut"),
])
def test_fixes_legacy_characters(broken: str, fixed: str):
    assert fix_legacy_romanian(broken) == fixed


def test_leaves_correct_text_alone():
    text = "Inima pompează sângele; în țesuturi și organe. Î, Â, î, â."
    assert fix_legacy_romanian(text) == text


def test_keep_text_changes_nothing():
    assert keep_text("Func]iile [1]") == "Func]iile [1]"
