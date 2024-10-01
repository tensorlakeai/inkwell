# pylint: disable=duplicate-code

import json
import logging
import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

from inkwell.io import read_image
from inkwell.table_extractor import TableExtractorFactory, TableExtractorType

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
        self._mock_json_table = {
            "header": ["Name", "Age", "City"],
            "data": [
                ["John Doe", "30", "New York"],
                ["Jane Doe", "25", "London"],
                ["Jim Beam", "50", "Paris"],
            ],
        }

    def _test_results(self, results):
        self.assertIsInstance(results, dict)

    def test_table_transformer_extractor(self):
        table_extractor = TableExtractorFactory.get_table_extractor(
            TableExtractorType.TABLE_TRANSFORMER
        )
        results = table_extractor.process(self._image_table)
        self._test_results(results)

    @patch(
        "inkwell.table_extractor.openai_4o_mini_table_extractor.OpenAI4OMiniOCR"  # pylint: disable=line-too-long
    )
    def test_openai_table_extractor(self, mock_openai_ocr):
        mock_client = MagicMock()
        mock_openai_ocr.return_value = mock_client
        mock_client.process.return_value = json.dumps(self._mock_json_table)

        table_extractor = TableExtractorFactory.get_table_extractor(
            TableExtractorType.OPENAI_GPT4O_MINI
        )
        results = table_extractor.process(self._image_table)
        self._test_results(results)
