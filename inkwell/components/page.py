from enum import Enum
from typing import List, Union

from PIL import Image
from pydantic import BaseModel

from inkwell.components.figure import Figure
from inkwell.components.layout import Layout
from inkwell.components.table import Table
from inkwell.components.text import TextBox


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
    layout: Layout
    page_image: Image.Image

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


class Document(BaseModel):
    """
    Document in a document.
    """

    pages: List[Page]
