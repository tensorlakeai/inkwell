import numpy as np

from inkwell.components import Layout, LayoutBlock, Rectangle
from inkwell.layout_detector.base import BaseLayoutDetector
from inkwell.layout_detector.layout_detector import LayoutDetectorType

try:
    from paddleocr import PPStructure
except ImportError as ex:
    raise ImportError(
        "paddleocr is not available. Please install it first."
    ) from ex


class PaddleDetector(BaseLayoutDetector):
    def __init__(self, **kwargs):
        super().__init__()
        self._label_map = {
            "text": "Text",
            "title": "Title",
            "list": "List",
            "table": "Table",
            "figure": "Figure",
            "footer": "Text",
            "header": "Text",
        }

        self._detection_threshold = kwargs.get("detection_threshold", 0.5)

        self._load_engine()

    def _load_engine(self):
        self._engine = PPStructure(
            table=True,
            ocr=True,
            show_log=False,
            layout_score_threshold=self._detection_threshold,
        )

    def _gather_output(self, results: dict):
        layout = Layout()
        for result in results:
            x_1, y_1, x_2, y_2 = result["bbox"]
            label = self._label_map[result["type"]]
            cur_block = LayoutBlock(
                Rectangle(x_1, y_1, x_2, y_2),
                type=label,
                score=result["score"],
            )
            layout.append(cur_block)
        return layout

    def process(self, image_batch: list[np.ndarray]) -> list[Layout]:
        results = self._engine(image_batch)
        return [self._gather_output(result) for result in results]

    @property
    def model_id(self) -> str:
        return LayoutDetectorType.PADDLE.value
