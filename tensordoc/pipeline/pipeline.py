import json
from typing import List

import numpy as np

from tensordoc.components import (
    Document,
    Image,
    Layout,
    Page,
    PageFragment,
    PageFragmentType,
    Table,
    TableEncoding,
    TextBox,
)
from tensordoc.io import convert_page_to_image, read_image, read_pdf_document
from tensordoc.layout_detector import LayoutDetectorFactory
from tensordoc.ocr import OCRFactory
from tensordoc.pipeline.pipeline_config import PipelineConfig
from tensordoc.table_detector import (
    TableDetectorFactory,
    TableExtractorFactory,
)


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self._initialize_layout_detector()
        self._initialize_ocr_detector()
        self._initialize_table_detector()
        self._initialize_table_segmentation_detector()

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

    def _process_tables(self, image: np.ndarray, layout: Layout):
        table_fragments = []
        for table_block in layout:
            table_image = table_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)

            if self.config.table_extractor:
                table_dict = self.table_extractor.process(table_image)
                table_text = json.dumps(table_dict, indent=4)
                table_encoding = TableEncoding.JSON
            else:
                table_text = self.ocr_detector.process(table_image)
                table_encoding = TableEncoding.TEXT
            table = Table(
                data=table_text,
                bbox=table_block.rectangle,
                score=table_block.score,
                image=table_image,
                encoding=table_encoding,
            )
            table_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TABLE,
                    content=table,
                )
            )
        return table_fragments

    def _process_figures(self, image: np.ndarray, layout: Layout):
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

    def _process_text(self, image: np.ndarray, layout: Layout):
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

    def process(
        self, document_path: str, pages_to_parse: List[int] = None
    ) -> Document:

        if self._is_native_pdf(document_path):
            document = self._read_pdf(document_path)
            pages = self._preprocess_native_pdf(document, pages_to_parse)
        else:
            pages = [(self._read_image(document_path), 0)]

        processed_pages = []
        print(f"Processing {len(pages)} pages")
        for page_image, page_number in pages:
            print(f"Processing page {page_number}")
            if self.layout_detector:
                layout = self.layout_detector.process(page_image)

                fragments = []
                figure_blocks = []
                table_blocks = []
                text_blocks = []

                for block in layout.get_blocks():
                    if block.type == "Figure":
                        figure_blocks.append(block)
                    elif block.type == "Table":
                        table_blocks.append(block)
                    else:
                        text_blocks.append(block)

                figure_fragments = self._process_figures(
                    page_image, figure_blocks
                )
                fragments.extend(figure_fragments)

                text_fragments = self._process_text(page_image, text_blocks)
                fragments.extend(text_fragments)

                if self.config.table_detector:
                    table_layout = self.table_detector.process(page_image)
                    table_fragments = self._process_tables(
                        page_image, table_layout
                    )
                    table_blocks = table_layout.get_blocks()
                else:
                    table_fragments = self._process_tables(
                        page_image, Layout(blocks=table_blocks)
                    )

                fragments.extend(table_fragments)
            else:
                print(
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
