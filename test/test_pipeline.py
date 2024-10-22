import logging
import os
import pickle
import unittest
from test.utils import (
    get_mock_figure_fragment,
    get_mock_table_fragment,
    get_mock_text_fragment,
)
from unittest.mock import patch

from inkwell.api.document import Document
from inkwell.pipeline import DefaultPipelineConfig, Pipeline

_PDF_URL = "https://pub-5dc4d0c0254749378ccbcfffa4bd2a1e.r2.dev/sample_ratings_report.pdf"  # noqa: E501, pylint: disable=line-too-long
_IMG_PATH = "./data/sample.png"

_logger = logging.getLogger(__name__)


class TestPipeline(unittest.TestCase):

    @staticmethod
    def _load_test_image_path():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, _IMG_PATH)
        return image_path

    def setUp(self):
        _logger.info("Running test: %s", self._testMethodName)
        self._pipeline = Pipeline(DefaultPipelineConfig())
        self._document_url = _PDF_URL
        self._image_path = self._load_test_image_path()

    @patch(
        "inkwell.pipeline.fragment_processor.FigureFragmentProcessor.process"
    )
    @patch("inkwell.pipeline.fragment_processor.TextFragmentProcessor.process")
    @patch(
        "inkwell.pipeline.fragment_processor.TableFragmentProcessor.process"
    )
    def test_process_pdf_document(
        self,
        mock_figure_fragment_processor,
        mock_text_fragment_processor,
        mock_table_fragment_processor,
    ):
        mock_figure_fragment_processor.return_value = (
            get_mock_figure_fragment()
        )
        mock_text_fragment_processor.return_value = get_mock_text_fragment()
        mock_table_fragment_processor.return_value = get_mock_table_fragment()
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
        self.assertIsNotNone(processed_document.pages[0].table_fragments())
        self.assertIsNotNone(processed_document.pages[0].figure_fragments())
        self.assertIsNotNone(processed_document.pages[0].text_fragments())

        try:
            pickle.dumps(processed_document)
            is_serializable = True
        except (pickle.PicklingError, TypeError):
            is_serializable = False

        self.assertTrue(
            is_serializable, "Processed document should be serializable"
        )
