import io
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from inkwell.api.document import Document
from inkwell.api.figure import Figure
from inkwell.api.page import PageFragment, PageFragmentType
from inkwell.api.table import Table, TableEncoding
from inkwell.api.text import TextBox
from inkwell.components import Layout
from inkwell.figure_extractor.base import BaseFigureExtractor
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
    def process(
        self, image: np.ndarray, layout: Union[Layout, None] = None
    ) -> List[PageFragment]:
        pass


class TableFragmentProcessor(FragmentProcessor):
    def __init__(
        self,
        ocr_detector: BaseOCR,
        table_extractor: Optional[BaseTableExtractor] = None,
    ):
        self.ocr_detector = ocr_detector
        self.table_extractor = table_extractor

    def process(
        self, image: np.ndarray, layout: Union[Layout, None] = None
    ) -> List[PageFragment]:
        _logger.info("Processing %d table fragments in page", len(layout))
        table_fragments = []
        table_images = []
        for table_block in layout.get_blocks():
            table_image = table_block.pad_ratio(0.05).crop_image(image)
            table_images.append(table_image)

        if self.table_extractor:
            table_data = self.table_extractor.process(table_images)
            table_encoding = TableEncoding.JSON
        else:
            table_data = self.ocr_detector.process(table_images)
            table_encoding = TableEncoding.TEXT

        for table_data, table_block, table_image in zip(
            table_data, layout.get_blocks(), table_images
        ):
            table_text = str(table_data)
            table_image_bytes = table_image.tobytes()
            table = Table(
                data=table_data,
                text=table_text,
                bbox=table_block.rectangle.bbox_dict(),
                score=table_block.score,
                image=Table.encode_image(table_image_bytes),
                encoding=table_encoding,
            )
            table_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TABLE,
                    content=table,
                    reading_order_index=table_block.reading_order_index,
                )
            )
        return table_fragments


class FigureFragmentProcessor(FragmentProcessor):
    def __init__(
        self, ocr_detector: BaseOCR, figure_extractor: BaseFigureExtractor
    ):
        self.ocr_detector = ocr_detector
        self.figure_extractor = figure_extractor

    def process(
        self, image: np.ndarray, layout: Union[Layout, None] = None
    ) -> List[PageFragment]:
        _logger.info("Processing %d figure fragments in page", len(layout))
        figure_fragments = []
        figure_images = []
        for figure_block in layout.get_blocks():
            figure_image = figure_block.pad_ratio(0.05).crop_image(image)
            figure_images.append(figure_image)

        if self.figure_extractor:
            ocr_results = self.figure_extractor.process(figure_images)
        else:
            ocr_results = self.ocr_detector.process(figure_images)

        for ocr_result, figure_block, figure_image in zip(
            ocr_results, layout.get_blocks(), figure_images
        ):
            figure_image = Image.fromarray(figure_image)
            img_bytes = io.BytesIO()
            figure_image.save(img_bytes, format="PNG")
            figure_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.FIGURE,
                    content=Figure(
                        image=Figure.encode_image(img_bytes.getvalue()),
                        bbox=figure_block.rectangle.bbox_dict(),
                        score=figure_block.score,
                        text=ocr_result,
                    ),
                    reading_order_index=figure_block.reading_order_index,
                )
            )
        return figure_fragments


class TextFragmentProcessor(FragmentProcessor):
    def __init__(self, ocr_detector: BaseOCR):
        self.ocr_detector = ocr_detector

    def process(
        self, image: np.ndarray, layout: Union[Layout, None] = None
    ) -> List[PageFragment]:
        num_fragments = 1 if layout is None else len(layout.get_blocks())
        _logger.info("Processing %d text fragments in page", num_fragments)

        if layout is None:
            ocr_results = self.ocr_detector.process([image])
            text_fragments = [
                PageFragment(
                    fragment_type=PageFragmentType.TEXT,
                    content=TextBox(
                        text=ocr_results[0],
                        text_type="text",
                        bbox=None,
                        score=None,
                        image=TextBox.encode_image(image.tobytes()),
                    ),
                    reading_order_index=None,
                )
            ]
            return text_fragments

        text_images = []
        text_fragments = []
        for text_block in layout:
            text_image = text_block.pad_ratio(0.05).crop_image(image)
            text_images.append(text_image)

        ocr_results = self.ocr_detector.process(text_images)
        for ocr_result, text_block, text_image in zip(
            ocr_results, layout.get_blocks(), text_images
        ):
            text_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TEXT,
                    content=TextBox(
                        text=ocr_result,
                        text_type=text_block.type,
                        bbox=text_block.rectangle.bbox_dict(),
                        score=text_block.score,
                        image=TextBox.encode_image(text_image.tobytes()),
                    ),
                    reading_order_index=text_block.reading_order_index,
                )
            )
        return text_fragments
