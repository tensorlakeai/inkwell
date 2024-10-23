import logging
from typing import List, Optional

from inkwell.api.document import Document
from inkwell.api.page import Page
from inkwell.components import Layout, LayoutBlock
from inkwell.figure_extractor import FigureExtractorFactory
from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.io import convert_page_to_image, read_image, read_pdf_document
from inkwell.layout_detector import LayoutDetectorFactory
from inkwell.layout_detector.base import BaseLayoutDetector
from inkwell.ocr import OCRFactory
from inkwell.ocr.base import BaseOCR
from inkwell.pipeline.fragment_processor import (
    FigureFragmentProcessor,
    TableFragmentProcessor,
    TextFragmentProcessor,
)
from inkwell.pipeline.pipeline_config import (
    DefaultPipelineConfig,
    PipelineConfig,
)
from inkwell.reading_order import ReadingOrderDetectorFactory
from inkwell.reading_order.base import BaseReadingOrderDetector
from inkwell.table_detector import TableDetectorFactory
from inkwell.table_detector.base import BaseTableDetector
from inkwell.table_extractor import TableExtractorFactory
from inkwell.table_extractor.base import BaseTableExtractor

_logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig = DefaultPipelineConfig(),
        layout_detector: BaseLayoutDetector = None,
        ocr_detector: BaseOCR = None,
        table_detector: BaseTableDetector = None,
        table_extractor: BaseTableExtractor = None,
        figure_extractor: BaseFigureExtractor = None,
        reading_order_detector: BaseReadingOrderDetector = None,
    ):
        self.config = config
        self._model_ids = {}

        self._initialize_layout_detector(layout_detector)
        self._initialize_ocr_detector(ocr_detector)
        self._initialize_table_detector(table_detector)
        self._initialize_table_extractor(table_extractor)
        self._initialize_figure_extractor(figure_extractor)
        self._initialize_reading_order_detector(reading_order_detector)
        self.table_fragment_processor = TableFragmentProcessor(
            ocr_detector=self.ocr_detector,
            table_extractor=self.table_extractor,
        )

        self.figure_fragment_processor = FigureFragmentProcessor(
            ocr_detector=self.ocr_detector,
            figure_extractor=self.figure_extractor,
        )

        self.text_fragment_processor = TextFragmentProcessor(
            ocr_detector=self.ocr_detector
        )

    def _initialize_layout_detector(
        self, layout_detector: Optional[BaseLayoutDetector] = None
    ):
        if layout_detector:
            self.layout_detector = layout_detector
        elif self.config.layout_detector:
            self.layout_detector = LayoutDetectorFactory.get_layout_detector(
                self.config.layout_detector,
                **self.config.layout_detector_kwargs,
            )
        else:
            self.layout_detector = None

        if self.layout_detector:
            self._model_ids["layout_detector"] = self.layout_detector.model_id

    def _initialize_ocr_detector(self, ocr_detector: Optional[BaseOCR] = None):
        if ocr_detector:
            self.ocr_detector = ocr_detector
        elif self.config.ocr_detector:
            self.ocr_detector = OCRFactory.get_ocr(
                self.config.ocr_detector,
                **{"inference_backend": self.config.inference_backend},
            )
        else:
            self.ocr_detector = None

        if self.ocr_detector:
            self._model_ids["ocr_detector"] = self.ocr_detector.model_id

    def _initialize_table_detector(
        self, table_detector: Optional[BaseTableDetector] = None
    ):
        if table_detector:
            self.table_detector = table_detector
        elif self.config.table_detector:
            self.table_detector = TableDetectorFactory.get_table_detector(
                self.config.table_detector, **self.config.table_detector_kwargs
            )
        else:
            self.table_detector = None

        if self.table_detector:
            self._model_ids["table_detector"] = self.table_detector.model_id

    def _initialize_table_extractor(
        self, table_extractor: Optional[BaseTableExtractor] = None
    ):
        if table_extractor:
            self.table_extractor = table_extractor
        elif self.config.table_extractor:
            self.table_extractor = TableExtractorFactory.get_table_extractor(
                self.config.table_extractor,
                **{"inference_backend": self.config.inference_backend},
            )
        else:
            self.table_extractor = None

        if self.table_extractor:
            self._model_ids["table_extractor"] = self.table_extractor.model_id

    def _initialize_figure_extractor(
        self, figure_extractor: Optional[BaseFigureExtractor] = None
    ):
        if figure_extractor:
            self.figure_extractor = figure_extractor
        elif self.config.figure_extractor:
            self.figure_extractor = (
                FigureExtractorFactory.get_figure_extractor(
                    self.config.figure_extractor
                )
            )
        else:
            self.figure_extractor = None

        if self.figure_extractor:
            self._model_ids["figure_extractor"] = (
                self.figure_extractor.model_id
            )

    def _initialize_reading_order_detector(
        self, reading_order_detector: Optional[BaseReadingOrderDetector] = None
    ):
        if reading_order_detector:
            self.reading_order_detector = reading_order_detector
        elif self.config.reading_order_detector:
            self.reading_order_detector = (
                ReadingOrderDetectorFactory.get_reading_order_detector(
                    self.config.reading_order_detector
                )
            )
        else:
            self.reading_order_detector = None

        if self.reading_order_detector:
            self._model_ids["reading_order_detector"] = (
                self.reading_order_detector.model_id
            )

    def _is_native_pdf(self, path: str) -> bool:
        return path.endswith(".pdf")

    def _read_pdf(self, path: str):
        return read_pdf_document(path)

    def _read_image(self, path: str):
        return read_image(path)

    def _preprocess_native_pdf(
        self, document: Document, pages_to_parse: List[int] = None
    ):
        pages = document.pages

        if pages_to_parse is not None:
            pages = [
                page for i, page in enumerate(pages) if i in pages_to_parse
            ]

        pages = [
            (convert_page_to_image(page), page.page_number) for page in pages
        ]

        return pages

    @staticmethod
    def _categorize_blocks(
        blocks: List[LayoutBlock],
    ) -> List[List[LayoutBlock]]:
        figure_blocks = []
        table_blocks = []
        text_blocks = []

        for block in blocks:
            if block.type == "Figure":
                figure_blocks.append(block)
            elif block.type == "Table":
                table_blocks.append(block)
            else:
                text_blocks.append(block)

        return figure_blocks, table_blocks, text_blocks

    def model_ids(self):
        return self._model_ids

    def _str_repr(self):
        config_str = "\nPipeline Configuration\n"
        for extractor, model_uid in self._model_ids.items():
            config_str += f"{extractor}: {model_uid}\n"
        return config_str

    def __repr__(self):
        return self._str_repr()

    def _get_pages(self, document_path: str, pages_to_parse: List[int] = None):
        if self._is_native_pdf(document_path):
            document = self._read_pdf(document_path)
            pages = self._preprocess_native_pdf(document, pages_to_parse)
        else:
            pages = [(self._read_image(document_path), 1)]
        return pages

    def process(
        self, document_path: str, pages_to_parse: List[int] = None
    ) -> Document:

        _logger.info(self._str_repr())

        pages = self._get_pages(document_path, pages_to_parse)

        processed_pages = []
        layout = None
        fragments = []
        for idx, (page_image, page_number) in enumerate(pages):
            _logger.info("Processing page %d/%d", idx + 1, len(pages))
            if self.layout_detector:
                layout = self.layout_detector.process([page_image])
                if self.reading_order_detector:
                    layout = self.reading_order_detector.process(
                        [page_image], layout
                    )

                layout = layout[0]
                figure_blocks, table_blocks, text_blocks = (
                    self._categorize_blocks(layout.get_blocks())
                )

                figure_fragments = self.figure_fragment_processor.process(
                    page_image, Layout(blocks=figure_blocks)
                )
                fragments.extend(figure_fragments)

                text_fragments = self.text_fragment_processor.process(
                    page_image, Layout(blocks=text_blocks)
                )
                fragments.extend(text_fragments)

                if self.config.table_detector:
                    table_layout = self.table_detector.process(page_image)
                    table_fragments = self.table_fragment_processor.process(
                        page_image, Layout(blocks=table_layout.get_blocks())
                    )
                else:
                    table_fragments = self.table_fragment_processor.process(
                        page_image, Layout(blocks=table_blocks)
                    )

                fragments.extend(table_fragments)
                layout = Layout(
                    blocks=figure_blocks + table_blocks + text_blocks
                )

            else:
                _logger.info(
                    "No layout detector configured, \
                    doing OCR on the whole image"
                )
            full_page_text_fragments = self.text_fragment_processor.process(
                page_image, None
            )
            fragments.extend(full_page_text_fragments)

            processed_pages.append(
                Page(
                    page_number=page_number,
                    page_fragments=fragments,
                    layout=layout.to_dict(),
                )
            )

        return Document(pages=processed_pages)
