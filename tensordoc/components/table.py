from enum import Enum

from pydantic import BaseModel


class TableEncoding(str, Enum):
    """
    Encoding of a table.
    """

    CSV = "csv"
    HTML = "html"
    JSON = "json"


class Table(BaseModel):
    """
    Table in a document.
    """

    data: str
    encoding: TableEncoding
