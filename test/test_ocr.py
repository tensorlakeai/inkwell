import os
import unittest

from tensordoc.ocr import OCRFactory, OCRType


class TestOCR(unittest.TestCase):
    def test_tesseract_ocr(self):
        ocr = OCRFactory.get_ocr(OCRType.TESSERACT, lang="eng")

        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")

        text = ocr.process(image_path)
        assert isinstance(text, str), "Text should be a string"
        assert len(text) > 0, "Text should not be empty"
        assert "receipt" in text, "Text should contain 'receipt'"
