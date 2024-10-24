import logging
from typing import List, Optional

from inkwell.api.document import Document
from inkwell.api.page import Page
from inkwell.figure_extractor import FigureExtractorFactory
from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.io import read_pdf_pages
from inkwell.layout_detector import LayoutDetectorFactory
from inkwell.layout_detector.base import BaseLayoutDetector
from inkwell.ocr import OCRFactory
from inkwell.ocr.base import BaseOCR
from inkwell.pipeline.fragment_processor import (
    FigureFragmentProcessor,
    TableFragmentProcessor,
    TextFragmentProcessor,
)
from inkwell.pipeline.layout_processor import LayoutProcessor
from inkwell.pipeline.pipeline_config import (
    DefaultPipelineConfig,
    PipelineConfig,
)
from inkwell.pipeline.utils import combine_fragments, split_layout_blocks
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

        self._layout_processor = LayoutProcessor(
            layout_detector=self.layout_detector,
            reading_order_detector=self.reading_order_detector,
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

    def model_ids(self):
        return self._model_ids

    def _str_repr(self):
        config_str = "\nPipeline Configuration\n"
        for extractor, model_uid in self._model_ids.items():
            config_str += f"{extractor}: {model_uid}\n"
        return config_str

    def __repr__(self):
        return self._str_repr()

    def process(
        self, document_path: str, pages_to_parse: List[int] = None
    ) -> Document:

        _logger.info(self._str_repr())

        pages = read_pdf_pages(document_path, pages_to_parse)
        pages_layouts = self._layout_processor.process(pages)

        document_page_blocks = split_layout_blocks(pages_layouts)

        text_fragments = self.text_fragment_processor.process(
            document_page_blocks
        )
        figure_fragments = self.figure_fragment_processor.process(
            document_page_blocks
        )
        table_fragments = self.table_fragment_processor.process(
            document_page_blocks
        )

        _logger.info("Combining fragments")
        pages_map = combine_fragments(
            figure_fragments, table_fragments, text_fragments
        )
        pages = []
        for page_number, page_fragments in pages_map.items():
            if self.reading_order_detector:
                page_fragments.sort(key=lambda x: x.reading_order_index)

            page = Page(
                page_number=page_number,
                page_fragments=page_fragments,
            )
            pages.append(page)

        document = Document(pages=pages)
        return document
