import os
import unittest

from tensordoc.io import read_image, read_pdf_as_images, read_pdf_document


class TestIO(unittest.TestCase):

    def setUp(self):
        curr_path = os.path.dirname(__file__)
        self._image_path = os.path.join(curr_path, "./data/sample.png")
        self._pdf_path = os.path.join(curr_path, "./data/sample_2.pdf")

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
