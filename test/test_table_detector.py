import os
from unittest import TestCase

import cv2

from tensordoc.table_detector import TableDetectorFactory, TableDetectorType


class TestTableDetector(TestCase):

    @staticmethod
    def load_test_image():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        return image

    def setUp(self):
        self._image = self.load_test_image()

    def test_get_table_detector(self):
        table_detector = TableDetectorFactory.get_table_detector(
            TableDetectorType.TABLE_TRANSFORMER
        )
        table_block = table_detector.process(self._image)

        table_blocks = [
            block for block in table_block if block.type == "table"
        ]

        assert table_blocks, "There should be at least one table block"
