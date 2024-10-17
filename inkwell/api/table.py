from base64 import b64encode
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


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
    bbox: Optional[dict[str, float]] = None
    text: Optional[str] = None
    score: Optional[float] = None
    image: Optional[str] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Table(bbox={self.bbox}, score={self.score})"

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Convert image bytes to base64 encoded string."""
        return b64encode(image_bytes).decode("utf-8")

    def to_dict(self) -> dict:
        return {
            "fragment_type": "table",
            "data": self.data,
            "encoding": self.encoding,
            "bbox": self.bbox,
            "text": self.text,
            "score": self.score,
        }
