from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from tensordoc.components.image import Image
from tensordoc.components.table import Table
from tensordoc.components.text import Text


class PageFragmentType(str, Enum):
    """
    Type of a page fragment.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class PageFragment(BaseModel):
    fragment_type: PageFragmentType
    text: Optional[Text] = None
    table: Optional[Table] = None
    figure: Optional[Image] = None


class Page(BaseModel):
    """
    Page in a document.
    """

    page_number: int
    fragments: List[PageFragment]


class Document(BaseModel):
    """
    Document in a document.
    """

    pages: List[Page]
