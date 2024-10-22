import logging
import os
import unittest

from inkwell.components.layout import Layout
from inkwell.io import read_image
from inkwell.layout_detector import LayoutDetectorFactory, LayoutDetectorType

_logger = logging.getLogger(__name__)


class TestLayoutDetector(unittest.TestCase):

    def setUp(self):
        _logger.info("Running test: %s", self._testMethodName)
        self.test_image = read_image(self.load_test_image())

    @staticmethod
    def check_detected_layout(layout: Layout):
        figures = [fig for fig in layout.get_blocks() if fig.type == "Figure"]
        tables = [
            table for table in layout.get_blocks() if table.type == "Table"
        ]
        texts = [text for text in layout.get_blocks() if text.type == "Text"]

        assert len(layout.get_blocks()) > 0
        assert len(figures) > 0 or len(tables) > 0
        assert len(texts) > 0

    @staticmethod
    def load_test_image():
        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        return image_path

    def test_faster_rcnn_layout_detector(self):

        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.FASTER_RCNN
        )

        layout = detector.process([self.test_image])[0]

        self.check_detected_layout(layout)

    def test_layoutlmv3_layout_detector(self):
        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.LAYOUTLMV3
        )

        layout = detector.process([self.test_image])[0]

        self.check_detected_layout(layout)

    def test_faster_rcnn_batch_predictor(self):

        test_images = [self.test_image] * 5
        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.FASTER_RCNN
        )

        layouts = detector.process(test_images)
        self.assertEqual(len(layouts), len(test_images))
        for layout in layouts:
            self.check_detected_layout(layout)


if __name__ == "__main__":
    unittest.main()
