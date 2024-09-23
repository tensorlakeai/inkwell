from typing import Optional

import numpy as np
from pydantic import BaseModel

from inkwell.components.elements import Rectangle


class TextBox(BaseModel):
    """
    Text in a document.
    """

    text: str
    text_type: Optional[str] = None
    bbox: Optional[Rectangle] = None
    score: Optional[float] = None
    image: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"TextBox(bbox={self.bbox}, score={self.score})"
