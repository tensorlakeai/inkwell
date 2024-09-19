from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel

from tensordoc.components.elements import Rectangle


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

    data: str
    encoding: TableEncoding
    bbox: Optional[Rectangle] = None
    text: Optional[str] = None
    score: Optional[float] = None
    image: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Table(bbox={self.bbox}, score={self.score})"
