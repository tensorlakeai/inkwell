from enum import Enum
from typing import List

import numpy as np
import pytesseract

from tensordoc.ocr.base import BaseOCR


class OCRType(Enum):
    TESSERACT = "tesseract"


class TesseractOCR(BaseOCR):

    def __init__(self, lang: str = "eng"):
        self._lang = lang

    def _detect(self, image: np.ndarray) -> List:
        results = {}
        text = pytesseract.image_to_string(image, lang=self._lang)
        results["text"] = text
        return results

    def process(self, image: np.ndarray) -> str:
        results = self._detect(image)
        return results["text"]


class OCRFactory:
    @staticmethod
    def get_ocr(ocr_type: OCRType, **kwargs):
        if ocr_type == OCRType.TESSERACT:
            return TesseractOCR(**kwargs)
        raise ValueError(f"Invalid OCR type: {ocr_type}")
