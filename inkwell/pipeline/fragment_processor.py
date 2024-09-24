from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from inkwell.components import (
    Document,
    Image,
    Layout,
    PageFragment,
    PageFragmentType,
    Table,
    TableEncoding,
    TextBox,
)
from inkwell.ocr.base import BaseOCR
from inkwell.table_detector.base import BaseTableExtractor


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
        table_fragments = []
        for table_block in layout:
            table_image = table_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)

            if self.table_extractor:
                table_data = self.table_extractor.process(table_image)
                table_encoding = TableEncoding.JSON
            else:
                table_data = self.ocr_detector.process(table_image)
                table_encoding = TableEncoding.TEXT

            table_text = str(table_data)
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
        figure_fragments = []
        for figure_block in layout:
            figure_image = figure_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            figure_text = self.ocr_detector.process(figure_image)
            figure_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.FIGURE,
                    content=Image(
                        image=figure_image,
                        bbox=figure_block.rectangle,
                        score=figure_block.score,
                        text=figure_text,
                    ),
                )
            )
        return figure_fragments


class TextFragmentProcessor(FragmentProcessor):
    def __init__(self, ocr_detector: BaseOCR):
        self.ocr_detector = ocr_detector

    def process(self, image: np.ndarray, layout: Layout) -> List[PageFragment]:
        text_fragments = []
        for text_block in layout:
            text_image = text_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            text_data = self.ocr_detector.process(text_image)
            text_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TEXT,
                    content=TextBox(
                        text=text_data,
                        text_type=text_block.type,
                        bbox=text_block.rectangle,
                        score=text_block.score,
                        image=text_image,
                    ),
                )
            )
        return text_fragments
