from pydantic import BaseModel


class Image(BaseModel):
    """
    Figure in a document.
    """

    data: bytes
    mime_type: str
