from typing import Optional

from PIL.Image import Image as PILImage
from pydantic import BaseModel

from inkwell.components.elements import Rectangle


class Image(BaseModel):
    """
    Figure in a document.
    """

    image: PILImage
    bbox: Optional[Rectangle] = None
    text: Optional[str] = None
    score: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Image(bbox={self.bbox}, score={self.score})"
