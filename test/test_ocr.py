import os
import unittest

from tensordoc.io import read_image
from tensordoc.ocr import OCRFactory, OCRType
import logging

_logger = logging.getLogger(__name__)

class TestOCR(unittest.TestCase):

    def setUp(self):
        _logger.debug(f"Running test: {self._testMethodName}")

    @staticmethod
    def _load_test_image():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        image = read_image(image_path)
        return image

    def test_tesseract_ocr(self):
        ocr = OCRFactory.get_ocr(OCRType.TESSERACT, lang="eng")

        image = self._load_test_image()

        text = ocr.process(image)
        assert isinstance(text, str), "Text should be a string"
        assert len(text) > 0, "Text should not be empty"
        assert "receipt" in text, "Text should contain 'receipt'"
