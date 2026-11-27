"""
Holds what the PDF looks like after it has been read:
its pages, the text blocks on each page, where each line sits,
whether a page was read as text or with OCR, and the page numbers.
 It's a clean, structured copy of the PDF's content

 With labelled boxes, each step knows exactly where to look and doesn't have to read the messy PDF again
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, PositiveInt, NonNegativeInt, model_validator

# Glues pages together into one long document text (used for page_offsets and chunking),
# so the last word of one page never sticks to the first word of the next.
PAGE_SEPARATOR = "\n\n"


class SourceRef(BaseModel):     #where the PDF lives
    uri: str
    allowed_groups: list[str] = ["*"] # Ex: cardiologie, sist_nervos, etc..

    model_config = ConfigDict(extra="forbid", frozen=True)

class Line(BaseModel):          # one line of text
    text: str                   #line's text
    y_top: float                  #recognise a line as sitting in the header or footer band
    y_bottom: float

    model_config = ConfigDict(extra="forbid", frozen=True)


class Block(BaseModel):         # one block of content
    kind: Literal[
        "paragraph",
        "heading",
        "subheading",
        "table",
    ]
    lines: list[Line]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PageText(BaseModel):      # one page
    page_index: PositiveInt
    page_label: str             # what's printed on the page ("22")
    extraction: Literal["text", "ocr"]
    blocks: list[Block]
    section: str = ""           # header description section

    model_config = ConfigDict(extra="forbid", frozen=True)

    def plain_text(self) -> str:
        """The page as one string: lines joined by a newline, blocks by a blank line.

        The ONE rule for page -> text. page_offsets and chunking must both use it,
        otherwise offsets would point to the wrong place.
        """
        return "\n\n".join("\n".join(line.text for line in block.lines) for block in self.blocks)


class OutlineEntry(BaseModel):
    level: PositiveInt
    title: str
    page_index: PositiveInt

    model_config = ConfigDict(extra="forbid", frozen=True)

class ParsedDocument(BaseModel): # The whole PDF Parsed
    doc_id: str                 # from ids.py
    title: str
    file_name: str
    outline: list[OutlineEntry]
    pages: list[PageText]
    page_offsets: list[NonNegativeInt] # list of small marks on that strip, one per page: "page 1 starts at letter 0, page 2 at letter 1,200, page 3 at letter 2,500".

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _check_page_offsets(self) -> "ParsedDocument":
        if len(self.page_offsets) != len(self.pages):
            raise ValueError("page_offsets must have one entry per page")
        if self.page_offsets and self.page_offsets[0] != 0:
            raise ValueError("page_offsets must start at 0")
        for previous, current in zip(self.page_offsets, self.page_offsets[1:]):
            if current < previous:
                raise ValueError("page_offsets must never decrease")
        return self


def compute_page_offsets(pages: list[PageText]) -> list[int]:
    """Where each page starts in the whole document's text.

    The document text is every page's plain_text() joined with PAGE_SEPARATOR,
    so each next page starts after the previous page's text plus the separator.
    Used by the parser, and again by cleaning after it changes the text.
    """
    offsets: list[int] = []
    position = 0
    for page in pages:
        offsets.append(position)
        position += len(page.plain_text()) + len(PAGE_SEPARATOR)
    return offsets
