from base64 import b64encode
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
    reading_order_index: Optional[int] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_type": self.fragment_type,
            "content": self.content.to_dict(),
            "reading_order_index": self.reading_order_index,
            "page_number": self.page_number,
        }


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    page_fragments: Optional[List[PageFragment]] = []
    layout: Optional[dict] = {}
    page_image: Optional[str] = None

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_fragments": [
                fragment.to_dict() for fragment in self.page_fragments
            ],
            "layout": self.layout,
            "page_image": self.page_image,
        }
