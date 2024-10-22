import numpy as np

from inkwell.components.layout import Layout
from inkwell.reading_order.base import BaseReadingOrderDetector


class NoOpReadingOrderDetector(BaseReadingOrderDetector):
    @property
    def model_id(self) -> str:
        return "no_op_reading_order_detector"

    def process(
        self, image_batch: list[np.ndarray], layout_batch: list[Layout]
    ) -> list[Layout]:
        _ = image_batch
        return layout_batch
