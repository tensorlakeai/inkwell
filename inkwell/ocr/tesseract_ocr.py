from typing import List

import numpy as np
import pytesseract

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType


class TesseractOCR(BaseOCR):

    def __init__(self, lang: str = "eng"):
        self._lang = lang

    @property
    def model_id(self) -> str:
        return OCRType.TESSERACT.value

    def _detect(self, image: np.ndarray) -> List:
        results = {}
        text = pytesseract.image_to_string(image, lang=self._lang)
        results["text"] = text
        return results

    def process(self, image: np.ndarray) -> str:
        results = self._detect(image)
        return results["text"]
