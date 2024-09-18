import os
import unittest

from tensordoc.components import Document
from tensordoc.pipeline import Pipeline

_PDF_URL = "https://pub-5dc4d0c0254749378ccbcfffa4bd2a1e.r2.dev/sample_ratings_report.pdf"  # noqa: E501, pylint: disable=line-too-long
_IMG_PATH = "/data/sample.png"


class TestPipeline(unittest.TestCase):

    @staticmethod
    def _load_test_image_path():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        return image_path

    def setUp(self):
        self._pipeline = Pipeline()
        self._document_url = _PDF_URL
        self._image_path = self._load_test_image_path()

    def test_process_pdf_document(self):
        processed_document = self._pipeline.process(
            self._document_url, pages_to_parse=[0]
        )
        self.assertIsNotNone(processed_document)
        self.assertIsInstance(processed_document, Document)
        self.assertEqual(len(processed_document.pages), 1)

    def test_process_image(self):
        processed_document = self._pipeline.process(self._image_path)
        self.assertIsNotNone(processed_document)
        self.assertIsInstance(processed_document, Document)
        self.assertEqual(len(processed_document.pages), 1)
        self.assertIsNotNone(processed_document.pages[0].get_table_fragments())
        self.assertIsNotNone(processed_document.pages[0].get_image_fragments())
        self.assertIsNotNone(processed_document.pages[0].get_text_fragments())
