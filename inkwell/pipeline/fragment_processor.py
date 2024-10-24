import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from inkwell.api.document import Document
from inkwell.api.figure import Figure
from inkwell.api.page import PageFragment, PageFragmentType
from inkwell.api.table import Table, TableEncoding
from inkwell.api.text import TextBox
from inkwell.components import LayoutBlock
from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.ocr.base import BaseOCR
from inkwell.pipeline.utils import DocumentPageBlocks
from inkwell.table_extractor.base import BaseTableExtractor

_logger = logging.getLogger(__name__)


class DocumentProcessor(ABC):
    @abstractmethod
    def process(
        self, document_path: str, pages_to_parse: Optional[list[int]] = None
    ) -> Document:
        pass


class FragmentProcessor(ABC):
    @abstractmethod
    def process(
        self, document_page_blocks: DocumentPageBlocks
    ) -> list[PageFragment]:
        pass


@dataclass
class TableFragmentInformation:
    page_number: int
    table_block: LayoutBlock


class TableFragmentProcessor(FragmentProcessor):
    def __init__(
        self,
        ocr_detector: BaseOCR,
        table_extractor: Optional[BaseTableExtractor] = None,
    ):
        self.ocr_detector = ocr_detector
        self.table_extractor = table_extractor

    def process(
        self, document_page_blocks: DocumentPageBlocks
    ) -> list[PageFragment]:
        table_images = []
        table_blocks: list[TableFragmentInformation] = []
        for page_block in document_page_blocks.page_blocks:
            for table_block in page_block.table_blocks:
                table_image = table_block.pad_ratio(0.05).crop_image(
                    page_block.page_image
                )
                table_images.append(table_image)
                table_blocks.append(
                    TableFragmentInformation(
                        page_number=page_block.page_number,
                        table_block=table_block,
                    )
                )

        if self.table_extractor:
            _logger.info(
                "Running table extractor on %d table fragments",
                len(table_images),
            )
            ocr_results = self.table_extractor.process(table_images)
            table_encoding = TableEncoding.JSON
        else:
            _logger.info(
                "Running OCR on %d table fragments", len(table_images)
            )
            ocr_results = self.ocr_detector.process(table_images)
            table_encoding = TableEncoding.TEXT

        table_fragments = []
        for ocr_result, table_block in zip(ocr_results, table_blocks):
            table_text = str(ocr_result)
            table = Table(
                data=ocr_result,
                text=table_text,
                bbox=table_block.table_block.rectangle.bbox_dict(),
                score=table_block.table_block.score,
                encoding=table_encoding,
            )
            table_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TABLE,
                    content=table,
                    reading_order_index=table_block.table_block.reading_order_index,
                    page_number=table_block.page_number,
                )
            )
        return table_fragments


@dataclass
class FigureFragmentInformation:
    page_number: int
    figure_block: LayoutBlock


class FigureFragmentProcessor(FragmentProcessor):
    def __init__(
        self, ocr_detector: BaseOCR, figure_extractor: BaseFigureExtractor
    ):
        self.ocr_detector = ocr_detector
        self.figure_extractor = figure_extractor

    def process(
        self, document_page_blocks: DocumentPageBlocks
    ) -> list[PageFragment]:
        figure_images = []
        figure_blocks: list[FigureFragmentInformation] = []
        for page_block in document_page_blocks.page_blocks:
            for figure_block in page_block.figure_blocks:
                figure_images.append(
                    figure_block.pad_ratio(0.05).crop_image(
                        page_block.page_image
                    )
                )
                figure_blocks.append(
                    FigureFragmentInformation(
                        page_number=page_block.page_number,
                        figure_block=figure_block,
                    )
                )

        _logger.info(
            "Running figure extractor on %d figure fragments",
            len(figure_images),
        )
        if self.figure_extractor:
            ocr_results = self.figure_extractor.process(figure_images)
        else:
            ocr_results = self.ocr_detector.process(figure_images)

        figure_fragments = []
        for ocr_result, figure_block in zip(ocr_results, figure_blocks):
            figure_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.FIGURE,
                    content=Figure(
                        bbox=figure_block.figure_block.rectangle.bbox_dict(),
                        score=figure_block.figure_block.score,
                        text=ocr_result,
                    ),
                    reading_order_index=figure_block.figure_block.reading_order_index,
                    page_number=figure_block.page_number,
                )
            )
        return figure_fragments


@dataclass
class TextFragmentInformation:
    page_number: int
    text_block: LayoutBlock


class TextFragmentProcessor(FragmentProcessor):
    def __init__(self, ocr_detector: BaseOCR):
        self.ocr_detector = ocr_detector

    def process(
        self, document_page_blocks: DocumentPageBlocks
    ) -> list[PageFragment]:
        text_images: list[np.ndarray] = []
        text_blocks: list[TextFragmentInformation] = []
        for page_block in document_page_blocks.page_blocks:
            for text_block in page_block.text_blocks:
                text_images.append(
                    text_block.pad_ratio(0.05).crop_image(
                        page_block.page_image
                    )
                )
                text_blocks.append(
                    TextFragmentInformation(
                        page_number=page_block.page_number,
                        text_block=text_block,
                    )
                )

        _logger.info("Running OCR on %d text fragments", len(text_images))
        ocr_results = self.ocr_detector.process(text_images)
        text_fragments = []
        for ocr_result, text_block in zip(ocr_results, text_blocks):
            text_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TEXT,
                    content=TextBox(
                        text=ocr_result,
                        text_type=text_block.text_block.type,
                        bbox=text_block.text_block.rectangle.bbox_dict(),
                        score=text_block.text_block.score,
                    ),
                    reading_order_index=text_block.text_block.reading_order_index,
                    page_number=text_block.page_number,
                )
            )
        return text_fragments
