import os
import unittest

from tensordoc.io import read_image, read_pdf_as_images, read_pdf_document

_PDF_URL = "https://pub-5dc4d0c0254749378ccbcfffa4bd2a1e.r2.dev/sample_ratings_report.pdf"  # noqa: E501, pylint: disable=line-too-long
_IMAGE_URL = "https://pub-5dc4d0c0254749378ccbcfffa4bd2a1e.r2.dev/sample.png"


class TestIO(unittest.TestCase):

    def setUp(self):
        curr_path = os.path.dirname(__file__)
        self._image_path = os.path.join(curr_path, "./data/sample.png")
        self._pdf_path = os.path.join(curr_path, "./data/sample_2.pdf")
        self._pdf_url = _PDF_URL
        self._image_url = _IMAGE_URL

    def test_read_image(self):
        image = read_image(self._image_path)
        self.assertIsNotNone(image)

    def test_read_pdf_document(self):
        document = read_pdf_document(self._pdf_path)
        self.assertIsNotNone(document)

    def test_read_pdf_as_images(self):
        images = read_pdf_as_images(self._pdf_path)
        self.assertIsNotNone(images)
        self.assertEqual(len(images), 5)

    def test_read_pdf_document_from_url(self):
        document = read_pdf_document(self._pdf_url)
        self.assertIsNotNone(document)

    def test_read_image_from_url(self):
        image = read_image(self._image_url)
        self.assertIsNotNone(image)
