import logging
import os
from unittest import TestCase

from inkwell.io import read_image
from inkwell.table_detector import (
    TableDetectorFactory,
    TableDetectorType,
    TableExtractorFactory,
    TableExtractorType,
)

_logger = logging.getLogger(__name__)


class TestTableDetector(TestCase):

    @staticmethod
    def load_test_image():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        image = read_image(image_path)
        return image

    @staticmethod
    def load_test_image_table():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample_table.png")
        image = read_image(image_path)
        return image

    def setUp(self):
        _logger.debug("Running test: %s", self._testMethodName)
        self._image = self.load_test_image()
        self._image_table = self.load_test_image_table()

    def test_get_table_detector(self):
        table_detector = TableDetectorFactory.get_table_detector(
            TableDetectorType.TABLE_TRANSFORMER
        )
        table_block = table_detector.process(self._image)

        table_blocks = [
            block for block in table_block if block.type == "table"
        ]

        assert table_blocks, "There should be at least one table block"

    def test_get_table_extractor(self):
        table_extractor = TableExtractorFactory.get_table_extractor(
            TableExtractorType.TABLE_TRANSFORMER
        )
        results = table_extractor.process(self._image_table)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results["header"]), 1)
        self.assertEqual(len(results["data"]), 5)
