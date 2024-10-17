from base64 import b64encode
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel

from inkwell.api.figure import Figure
from inkwell.api.table import Table
from inkwell.api.text import TextBox


class PageFragmentType(str, Enum):
    """
    Type of a page fragment.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class PageFragment(BaseModel):
    fragment_type: PageFragmentType
    content: Union[TextBox, Table, Figure]


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    page_fragments: List[PageFragment]
    layout: Optional[dict]
    page_image: str

    def text_fragments(self) -> List[TextBox]:
        return [
            fragment
            for fragment in self.page_fragments
            if fragment.fragment_type == PageFragmentType.TEXT
        ]

    def table_fragments(self) -> List[Table]:
        return [
            fragment
            for fragment in self.page_fragments
            if fragment.fragment_type == PageFragmentType.TABLE
        ]

    def figure_fragments(self) -> List[Figure]:
        return [
            fragment
            for fragment in self.page_fragments
            if fragment.fragment_type == PageFragmentType.FIGURE
        ]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Convert image bytes to base64 encoded string."""
        return b64encode(image_bytes).decode("utf-8")


class Document(BaseModel):
    """
    Document in a document.
    """

    pages: List[Page]
