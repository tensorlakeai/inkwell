from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from inkwell.components import Layout, LayoutBlock, Rectangle
from inkwell.ocr import OCRFactory, OCRType
from inkwell.ocr.base import BaseOCR
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import (
    TABLE_TRANSFORMER_TABLE_EXTRACTOR_CONFIG,
)
from inkwell.table_extractor.table_extractor import TableExtractorType


class TableTransformerExtractor(BaseTableExtractor):
    def __init__(self, ocr_detector: Optional[BaseOCR] = None):
        self._feature_extractor = DetrImageProcessor()
        self._load_processor()
        self._load_model()
        if ocr_detector:
            self._ocr_detector = ocr_detector
        else:
            self._load_ocr_detector()

    @property
    def model_id(self) -> str:
        return TableExtractorType.TABLE_TRANSFORMER.value

    def _load_processor(self):
        self._processor = DetrImageProcessor()

    def _load_model(self):
        self._model = TableTransformerForObjectDetection.from_pretrained(
            TABLE_TRANSFORMER_TABLE_EXTRACTOR_CONFIG.model_name_hf
        )

    def _load_ocr_detector(self):
        self._ocr_detector = OCRFactory.get_ocr(OCRType.TESSERACT)

    def _get_cell_coordinates_by_row(self, table_blocks: list[LayoutBlock]):

        rows: list[LayoutBlock] = []
        columns: list[LayoutBlock] = []

        for block in table_blocks:
            if block.type in ["table row", "table column header"]:
                rows.append(block)
            elif block.type == "table column":
                columns.append(block)

        rows.sort(key=lambda x: x.rectangle.y_1)
        columns.sort(key=lambda x: x.rectangle.x_1)

        def find_cell_coordinates(row: LayoutBlock, column: LayoutBlock):
            cell_bbox = [
                column.rectangle.x_1,
                row.rectangle.y_1,
                column.rectangle.x_2,
                row.rectangle.y_2,
            ]
            return cell_bbox

        cell_coordinates_for_rows = []
        for row in rows:
            row_information = {}
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append(
                    {"column": column.rectangle, "cell": cell_bbox}
                )

            row_cells.sort(key=lambda x: x["column"].x_1)

            row_information["row"] = row.rectangle
            row_information["cells"] = row_cells
            row_information["row_type"] = row.type

            cell_coordinates_for_rows.append(row_information)

        return cell_coordinates_for_rows

    def _return_cell_results(
        self, image: np.ndarray, table_segementation_layout: Layout
    ) -> dict:
        table_blocks = table_segementation_layout.get_blocks()
        cell_coordinates_for_rows = self._get_cell_coordinates_by_row(
            table_blocks
        )

        results = defaultdict(list)
        for row in cell_coordinates_for_rows:

            row_text = []
            for cell in row["cells"]:
                cell_block = LayoutBlock(
                    block=Rectangle(
                        x_1=cell["cell"][0],
                        y_1=cell["cell"][1],
                        x_2=cell["cell"][2],
                        y_2=cell["cell"][3],
                    ),
                    text="",
                )
                cell_block_image = cell_block.pad(10, 10, 10, 10).crop_image(
                    image
                )
                result = self._ocr_detector.process([cell_block_image])[
                    0
                ].strip()

                row_text.append(result)

            if row["row_type"] == "table column header":
                results["header"].append(row_text)
            else:
                results["data"].append(row_text)

        return dict(results)

    def _convert_to_rows_cols(self, outputs: dict) -> Layout:
        blocks = []
        scores = outputs["scores"]
        labels = outputs["labels"]
        boxes = outputs["boxes"]

        total = len(scores)
        for i in range(total):

            block = LayoutBlock(
                text="",
                block=Rectangle(
                    x_1=boxes[i][0],
                    y_1=boxes[i][1],
                    x_2=boxes[i][2],
                    y_2=boxes[i][3],
                ),
                score=scores[i].item(),
                type=self._model.config.id2label[labels[i].item()],
            )
            blocks.append(block)

        segmented_table_layout = Layout(
            blocks=[b for b in blocks if b.type != "table"]
        )

        return segmented_table_layout

    def _return_row_results(
        self, image: np.ndarray, table_segementation_layout: Layout
    ):
        table_blocks = table_segementation_layout.get_blocks()

        table_dict = defaultdict(list)

        for block in table_blocks:
            if block.type in [
                "table row",
                "table column header",
                "table column",
            ]:
                row_image = block.pad(10, 10, 10, 10).crop_image(image)
                result = self._ocr_detector.process([row_image])[0].strip()
                if block.type == "table column header":
                    table_dict["header"].append(result)
                elif block.type == "table row":
                    table_dict["rows"].append(result)
                else:
                    table_dict["columns"].append(result)
        return table_dict

    def process(self, image_batch: list[np.ndarray]) -> list[dict]:

        table_results = []
        for img in tqdm(image_batch, desc="Processing table fragments"):
            image_pil = Image.fromarray(img)
            encoding = self._processor(image_pil, return_tensors="pt")

            with torch.no_grad():
                outputs = self._model(**encoding)

            target_sizes = [image_pil.size[::-1]]
            results = self._processor.post_process_object_detection(
                outputs, threshold=0.6, target_sizes=target_sizes
            )[0]

            table_segmentation_layout = self._convert_to_rows_cols(results)

            table_dict = self._return_row_results(
                img, table_segmentation_layout
            )
            table_results.append(table_dict)

        return table_results
