""" Errors raised while extracting text from a PDF. All of them inherit from ExtractionFailed,
 so the indexing pipeline can catch one type per file, record the reason in the report,
 and continue with the next PDF """

class ExtractionFailed(Exception):
    """PDF could not be turned into a ParsedDocument.
    Carries a short human-readable reason on the file name. Raised directly for unexpected problems"""

    def __init__(self, name: str, reason: str):
        super().__init__(f"{name}: {reason}")
        self.name = name
        self.reason = reason

class CorruptPdf(ExtractionFailed):
    """ The PDF is damaged or not a Valid PDF so it cannot be opened or read"""

    def __init__(self, name: str):
        super().__init__(name,  "PDF could not be opened or read")


