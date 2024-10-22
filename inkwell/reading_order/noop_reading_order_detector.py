from typing import List, Union

from inkwell.components.layout import Layout
from inkwell.reading_order.base import BaseReadingOrderDetector


class NoOpReadingOrderDetector(BaseReadingOrderDetector):
    @property
    def model_id(self) -> str:
        return "no_op_reading_order_detector"

    def process(
        self, layout: Union[Layout, List[Layout]]
    ) -> Union[Layout, List[Layout]]:
        return layout
