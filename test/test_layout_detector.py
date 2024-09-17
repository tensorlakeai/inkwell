import os
import unittest

from tensordoc.io import read_image
from tensordoc.layout_detector import LayoutDetectorFactory, LayoutDetectorType


class TestLayoutDetector(unittest.TestCase):

    def setUp(self):
        self.test_image = read_image(self.load_test_image())

    @staticmethod
    def check_detected_layout(layout):
        figures = [fig for fig in layout if fig.type == "Figure"]
        texts = [text for text in layout if text.type == "Text"]

        assert len(layout) > 0
        assert len(figures) > 0
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

        layout = detector.process(self.test_image)

        self.check_detected_layout(layout)

    def test_dit_layout_detector(self):
        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.DIT
        )

        layout = detector.process(self.test_image)

        self.check_detected_layout(layout)

    def test_layoutlmv3_layout_detector(self):
        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.LAYOUTLMV3
        )

        layout = detector.process(self.test_image)

        self.check_detected_layout(layout)


if __name__ == "__main__":
    unittest.main()
