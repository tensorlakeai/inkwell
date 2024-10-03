import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage

from inkwell.components import (
    Document,
    Figure,
    Layout,
    PageFragment,
    PageFragmentType,
    Table,
    TableEncoding,
    TextBox,
)
from inkwell.ocr.base import BaseOCR
from inkwell.table_extractor.base import BaseTableExtractor

_logger = logging.getLogger(__name__)


class DocumentProcessor(ABC):
    @abstractmethod
    def process(
        self, document_path: str, pages_to_parse: Optional[List[int]] = None
    ) -> Document:
        pass


class FragmentProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, layout: Layout) -> List[PageFragment]:
        pass


class TableFragmentProcessor(FragmentProcessor):
    def __init__(
        self,
        ocr_detector: BaseOCR,
        table_extractor: Optional[BaseTableExtractor] = None,
    ):
        self.ocr_detector = ocr_detector
        self.table_extractor = table_extractor

    def process(self, image: np.ndarray, layout: Layout) -> List[PageFragment]:
        _logger.info("Processing %d table fragments in page", len(layout))
        table_fragments = []
        table_images = []
        for table_block in layout:
            table_image = table_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            table_images.append(table_image)

        if self.table_extractor:
            table_data = self.table_extractor.process(table_images)
            table_encoding = TableEncoding.JSON
        else:
            table_data = self.ocr_detector.process(table_images)
            table_encoding = TableEncoding.TEXT

        for table_data, table_block, table_image in zip(
            table_data, layout, table_images
        ):
            table_text = str(table_data)
            table_image = PILImage.fromarray(table_image)
            table = Table(
                data=table_data,
                text=table_text,
                bbox=table_block.rectangle,
                score=table_block.score,
                image=table_image,
                encoding=table_encoding,
            )
            table_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TABLE, content=table
                )
            )
        return table_fragments


class FigureFragmentProcessor(FragmentProcessor):
    def __init__(self, ocr_detector: BaseOCR):
        self.ocr_detector = ocr_detector

    def process(self, image: np.ndarray, layout: Layout) -> List[PageFragment]:
        _logger.info("Processing %d figure fragments in page", len(layout))
        figure_fragments = []
        figure_images = []
        for figure_block in layout:
            figure_image = figure_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            figure_images.append(figure_image)

        ocr_results = self.ocr_detector.process(figure_images)
        for ocr_result, figure_block, figure_image in zip(
            ocr_results, layout, figure_images
        ):
            figure_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.FIGURE,
                    content=Figure(
                        image=PILImage.fromarray(figure_image),
                        bbox=figure_block.rectangle,
                        score=figure_block.score,
                        text=ocr_result,
                    ),
                )
            )
        return figure_fragments


class TextFragmentProcessor(FragmentProcessor):
    def __init__(self, ocr_detector: BaseOCR):
        self.ocr_detector = ocr_detector

    def process(self, image: np.ndarray, layout: Layout) -> List[PageFragment]:
        _logger.info("Processing %d text fragments in page", len(layout))
        text_fragments = []
        text_images = []
        for text_block in layout:
            text_image = text_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            text_images.append(text_image)

        ocr_results = self.ocr_detector.process(text_images)
        for ocr_result, text_block, text_image in zip(
            ocr_results, layout, text_images
        ):
            text_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TEXT,
                    content=TextBox(
                        text=ocr_result,
                        text_type=text_block.type,
                        bbox=text_block.rectangle,
                        score=text_block.score,
                        image=PILImage.fromarray(text_image),
                    ),
                )
            )
        return text_fragments
