from pydantic import BaseModel


class Text(BaseModel):
    """
    Text in a document.
    """

    data: str
