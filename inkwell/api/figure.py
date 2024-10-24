from base64 import b64encode
from typing import Optional

from pydantic import BaseModel


class Figure(BaseModel):
    """
    Figure in a document.
    """

    image: Optional[str] = None
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
        return b64encode(image_bytes)

    def to_dict(self) -> dict:
        return {
            "fragment_type": "figure",
            "image": self.image,
            "bbox": self.bbox,
            "text": self.text,
            "score": self.score,
        }
