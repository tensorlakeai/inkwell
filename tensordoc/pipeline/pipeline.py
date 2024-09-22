import logging
from typing import List

from tensordoc.components import (
    Document,
    Layout,
    LayoutBlock,
    Page,
    PageFragment,
    PageFragmentType,
)
from tensordoc.io import convert_page_to_image, read_image, read_pdf_document
from tensordoc.layout_detector import LayoutDetectorFactory
from tensordoc.layout_detector.base import BaseLayoutDetector
from tensordoc.ocr import OCRFactory
from tensordoc.ocr.base import BaseOCR
from tensordoc.pipeline.fragment_processor import (
    FigureFragmentProcessor,
    TableFragmentProcessor,
    TextFragmentProcessor,
)
from tensordoc.pipeline.pipeline_config import PipelineConfig
from tensordoc.table_detector import (
    TableDetectorFactory,
    TableExtractorFactory,
)
from tensordoc.table_detector.base import BaseTableDetector, BaseTableExtractor

_logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig = PipelineConfig(),
        layout_detector: BaseLayoutDetector = None,
        ocr_detector: BaseOCR = None,
        table_detector: BaseTableDetector = None,
        table_extractor: BaseTableExtractor = None,
    ):

        self.config = config

        self._initialize_layout_detector()
        self._initialize_ocr_detector()
        self._initialize_table_detector()
        self._initialize_table_segmentation_detector()

        # Passing the custom components replaces the default components
        if layout_detector:
            self.layout_detector = layout_detector
        if ocr_detector:
            self.ocr_detector = ocr_detector
        if table_detector:
            self.table_detector = table_detector
        if table_extractor:
            self.table_extractor = table_extractor

        self.table_fragment_processor = TableFragmentProcessor(
            ocr_detector=self.ocr_detector,
            table_extractor=self.table_extractor,
        )

        self.figure_fragment_processor = FigureFragmentProcessor(
            ocr_detector=self.ocr_detector
        )

        self.text_fragment_processor = TextFragmentProcessor(
            ocr_detector=self.ocr_detector
        )

    def _initialize_layout_detector(self):
        if self.config.layout_detector:
            self.layout_detector = LayoutDetectorFactory.get_layout_detector(
                self.config.layout_detector,
                **self.config.layout_detector_kwargs,
            )

    def _initialize_ocr_detector(self):
        if self.config.ocr_detector:
            self.ocr_detector = OCRFactory.get_ocr(self.config.ocr_detector)

    def _initialize_table_detector(self):
        if self.config.table_detector:
            self.table_detector = TableDetectorFactory.get_table_detector(
                self.config.table_detector, **self.config.table_detector_kwargs
            )

    def _initialize_table_segmentation_detector(self):
        if self.config.table_extractor:
            self.table_extractor = TableExtractorFactory.get_table_extractor(
                self.config.table_extractor
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

    def process(
        self, document_path: str, pages_to_parse: List[int] = None
    ) -> Document:

        if self._is_native_pdf(document_path):
            document = self._read_pdf(document_path)
            pages = self._preprocess_native_pdf(document, pages_to_parse)
        else:
            pages = [(self._read_image(document_path), 0)]

        processed_pages = []
        for page_image, page_number in pages:
            _logger.info("Processing page %d/%d", page_number, len(pages))
            if self.layout_detector:
                layout = self.layout_detector.process(page_image)

                fragments = []
                figure_blocks, table_blocks, text_blocks = (
                    self._categorize_blocks(layout.get_blocks())
                )

                figure_fragments = self.figure_fragment_processor.process(
                    page_image, figure_blocks
                )
                fragments.extend(figure_fragments)

                text_fragments = self.text_fragment_processor.process(
                    page_image, text_blocks
                )
                fragments.extend(text_fragments)

                if self.config.table_detector:
                    table_layout = self.table_detector.process(page_image)
                    table_fragments = self.table_fragment_processor.process(
                        page_image, table_layout
                    )
                else:
                    table_fragments = self.table_fragment_processor.process(
                        page_image, Layout(blocks=table_blocks)
                    )

                fragments.extend(table_fragments)
            else:
                _logger.info(
                    "No layout detector configured, \
                    doing OCR on the whole image"
                )
                fragments.append(
                    PageFragment(
                        fragment_type=PageFragmentType.TEXT,
                        content=self.ocr_detector.process(page_image),
                    )
                )

            layout = Layout(blocks=figure_blocks + table_blocks + text_blocks)

            processed_pages.append(
                Page(
                    page_number=page_number,
                    page_fragments=fragments,
                    layout=layout,
                )
            )

        return Document(pages=processed_pages)
