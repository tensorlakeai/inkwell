from enum import Enum
from typing import Optional, Union

from PIL.Image import Image as PILImage
from pydantic import BaseModel

from inkwell.components.elements import Rectangle


class TableEncoding(str, Enum):
    """
    Encoding of a table.
    """

    CSV = "csv"
    HTML = "html"
    JSON = "json"
    TEXT = "text"
    DICT = "dict"


class Table(BaseModel):
    """
    Table in a document.
    """

    data: Union[dict, str]
    encoding: TableEncoding
    bbox: Optional[Rectangle] = None
    text: Optional[str] = None
    score: Optional[float] = None
    image: Optional[PILImage] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Table(bbox={self.bbox}, score={self.score})"
