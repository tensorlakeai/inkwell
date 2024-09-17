from typing import Optional

import numpy as np
from pydantic import BaseModel

from tensordoc.components.elements import Rectangle


class Image(BaseModel):
    """
    Figure in a document.
    """

    image: np.ndarray
    bbox: Optional[Rectangle] = None
    text: Optional[str] = None
    score: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Image(bbox={self.bbox}, score={self.score})"
