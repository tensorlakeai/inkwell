from typing import List

import numpy as np

from tensordoc.components import (
    Document,
    Image,
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
    TableSegmentationFactory,
)


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self._initialize_layout_detector()
        self._initialize_ocr_detector()
        self._initialize_table_detector()
        self._initialize_table_segmentation_detector()

    def _initialize_layout_detector(self):
        self.layout_detector = LayoutDetectorFactory.get_layout_detector(
            self.config.layout_detector, **self.config.layout_detector_kwargs
        )

    def _initialize_ocr_detector(self):
        self.ocr_detector = OCRFactory.get_ocr(self.config.ocr_detector)

    def _initialize_table_detector(self):
        self.table_detector = TableDetectorFactory.get_table_detector(
            self.config.table_detector
        )

    def _initialize_table_segmentation_detector(self):
        self.table_segmentation_detector = (
            TableSegmentationFactory.get_table_segmentation(
                self.config.table_segmentation_detector
            )
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

    def _process_tables(self, image: np.ndarray):
        layout = self.table_detector.process(image)
        table_fragments = []
        for table_block in layout:
            table_image = table_block.pad(
                left=5, right=5, top=5, bottom=5
            ).crop_image(image)
            table_text = self.ocr_detector.process(table_image)
            table = Table(
                data=table_text,
                bbox=table_block.rectangle,
                score=table_block.score,
                image=table_image,
                encoding=TableEncoding.TEXT,
            )
            table_fragments.append(
                PageFragment(
                    fragment_type=PageFragmentType.TABLE,
                    content=table,
                )
            )
        return table_fragments

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
                detected_block_types = set(block.type for block in layout)

                fragments = []
                for block in layout.get_blocks():

                    if block.type == "Figure":
                        block_image = block.pad(
                            left=5, right=5, top=5, bottom=5
                        ).crop_image(page_image)
                        image_text = self.ocr_detector.process(block_image)
                        fragments.append(
                            PageFragment(
                                fragment_type=PageFragmentType.FIGURE,
                                content=Image(
                                    image=block_image,
                                    bbox=block.rectangle,
                                    score=block.score,
                                    text=image_text,
                                ),
                            )
                        )

                    elif block.type in detected_block_types - {
                        "Figure",
                        "Table",
                    }:
                        block_image = block.pad(
                            left=5, right=5, top=5, bottom=5
                        ).crop_image(page_image)
                        text_data = self.ocr_detector.process(block_image)
                        fragments.append(
                            PageFragment(
                                fragment_type=PageFragmentType.TEXT,
                                content=TextBox(
                                    text=text_data,
                                    bbox=block.rectangle,
                                    score=block.score,
                                    image=block_image,
                                ),
                            )
                        )
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
            if self.config.table_detector:
                table_fragments = self._process_tables(page_image)
                fragments.extend(table_fragments)
            processed_pages.append(
                Page(
                    page_number=page_number,
                    page_fragments=fragments,
                    layout=layout,
                )
            )

        return Document(pages=processed_pages)
