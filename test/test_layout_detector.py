import os
import unittest

import cv2

from tensordoc.layout_detector import LayoutDetectorFactory, LayoutDetectorType


class TestLayoutDetector(unittest.TestCase):
    def test_faster_rcnn_layout_detector(self):
        detector = LayoutDetectorFactory.get_layout_detector(
            LayoutDetectorType.FASTER_RCNN
        )

        curr_path = os.path.dirname(__file__)
        image_path = os.path.join(curr_path, "./data/sample.png")
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        layout = detector.process(image)

        tables = [tab for tab in layout if tab.type == "Table"]
        figures = [fig for fig in layout if fig.type == "Figure"]
        lists = [lst for lst in layout if lst.type == "List"]
        titles = [title for title in layout if title.type == "Title"]
        texts = [text for text in layout if text.type == "Text"]

        assert len(layout) > 0
        assert len(tables) > 0
        assert len(figures) > 0
        assert len(lists) > 0
        assert len(titles) > 0
        assert len(texts) > 0


if __name__ == "__main__":
    unittest.main()
