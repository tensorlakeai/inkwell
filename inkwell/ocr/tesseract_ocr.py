from typing import List, Optional, Union

import numpy as np
import pytesseract

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType


class TesseractOCR(BaseOCR):

    def __init__(self, **kwargs):
        self._lang = kwargs.get("lang", "eng")

    @property
    def model_id(self) -> str:
        return OCRType.TESSERACT.value

    def _detect(self, image: np.ndarray) -> str:
        text = pytesseract.image_to_string(image, lang=self._lang)
        return text

    def process(
        self,
        image: Union[np.ndarray, List[np.ndarray]],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """
        Processes the image(s) and returns the text(s) detected.

        Args:
            image (np.ndarray or list[np.ndarray]): The image(s) to process.

        Returns:
            str or list[str]: The text(s) detected.
        """
        _, _ = user_prompt, system_prompt
        if isinstance(image, list):
            return [self._detect(img) for img in image]
        return self._detect(image)
