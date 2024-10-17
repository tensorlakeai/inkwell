from base64 import b64encode
from typing import Optional

from pydantic import BaseModel, Field


class Figure(BaseModel):
    """
    Figure in a document.
    """

    image: str = Field(exclude=True)
    bbox: Optional[dict[str, float]] = None
    text: Optional[str] = None
    score: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Figure(bbox={self.bbox}, score={self.score})"

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Convert image bytes to base64 encoded string."""
        return b64encode(image_bytes).decode("utf-8")
