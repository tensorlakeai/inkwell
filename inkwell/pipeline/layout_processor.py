import logging
from typing import Optional

from inkwell.components.document import PageImage
from inkwell.layout_detector.base import BaseLayoutDetector
from inkwell.reading_order.base import BaseReadingOrderDetector

_logger = logging.getLogger(__name__)


class LayoutProcessor:
    def __init__(
        self,
        layout_detector: BaseLayoutDetector,
        reading_order_detector: Optional[BaseReadingOrderDetector] = None,
    ):

        self._layout_detector = layout_detector
        self._reading_order_detector = reading_order_detector

    def process(self, page_images: list[PageImage]) -> list[PageImage]:
        _logger.info("Running layout detector on %d pages", len(page_images))
        image_batch = [page_image.page_image for page_image in page_images]
        layouts = self._layout_detector.process(image_batch=image_batch)
        if self._reading_order_detector:
            layouts = self._reading_order_detector.process(
                image_batch=image_batch, layout_batch=layouts
            )

        return [
            PageImage(
                page_image=page_image.page_image,
                page_number=page_image.page_number,
                page_layout=layout,
            )
            for (page_image, layout) in zip(page_images, layouts)
        ]
