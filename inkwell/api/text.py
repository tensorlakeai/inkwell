from base64 import b64encode
from typing import Optional

from pydantic import BaseModel, Field


class TextBox(BaseModel):
    """
    Text in a document.
    """

    text: str
    text_type: Optional[str] = None
    bbox: Optional[dict[str, float]] = None
    score: Optional[float] = None
    image: Optional[str] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"TextBox(bbox={self.bbox}, score={self.score})"

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Convert image bytes to base64 encoded string."""
        return b64encode(image_bytes).decode("utf-8")

    def to_dict(self) -> dict:
        return {
            "fragment_type": "text",
            "text": self.text,
            "text_type": self.text_type,
            "bbox": self.bbox,
            "score": self.score,
        }
