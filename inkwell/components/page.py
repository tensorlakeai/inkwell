from enum import Enum
from typing import List, Union

from pydantic import BaseModel

from inkwell.components.image import Image
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
    content: Union[TextBox, Table, Image]


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    page_fragments: List[PageFragment]
    layout: Layout

    def get_text_fragments(self) -> List[TextBox]:
        return [
            fragment
            for fragment in self.page_fragments
            if fragment.fragment_type == PageFragmentType.TEXT
        ]

    def get_table_fragments(self) -> List[Table]:
        return [
            fragment
            for fragment in self.page_fragments
            if fragment.fragment_type == PageFragmentType.TABLE
        ]

    def get_image_fragments(self) -> List[Image]:
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
