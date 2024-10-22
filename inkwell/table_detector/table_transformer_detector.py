from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from inkwell.components import Layout, LayoutBlock, Rectangle
from inkwell.table_detector.base import BaseTableDetector
from inkwell.table_detector.config import (
    TABLE_TRANSFORMER_TABLE_DETECTOR_CONFIG,
)
from inkwell.table_detector.table_detector import TableDetectorType


class TableTransformerDetector(BaseTableDetector):
    def __init__(self, **kwargs):
        self._feature_extractor = DetrImageProcessor()
        self._cfg = TABLE_TRANSFORMER_TABLE_DETECTOR_CONFIG
        self._load_processor()
        self._load_model()

        self._detection_threshold = kwargs.get("detection_threshold", 0.5)

    @property
    def model_id(self) -> str:
        return TableDetectorType.TABLE_TRANSFORMER.value

    def _load_model(self):
        self._model = TableTransformerForObjectDetection.from_pretrained(
            self._cfg.model_name_hf
        )

    def _load_processor(self):
        self._processor = DetrImageProcessor()

    def _post_process_results(self, outputs: Dict) -> Layout:

        table_idx = outputs["labels"] == 0

        table_scores = outputs["scores"][table_idx]
        table_labels = outputs["labels"][table_idx]
        table_boxes = outputs["boxes"][table_idx]

        blocks = []
        for score, label, box in zip(table_scores, table_labels, table_boxes):
            block = LayoutBlock(
                text="",
                block=Rectangle(
                    x_1=box[0], y_1=box[1], x_2=box[2], y_2=box[3]
                ),
                score=score.item(),
                type=self._model.config.id2label[label.item()],
            )
            blocks.append(block)

        table_block = Layout(blocks=blocks)
        return table_block

    def process(self, image: list[np.ndarray]) -> list[Layout]:
        return [self._process_image(img) for img in image]

    def _process_image(self, image: np.ndarray) -> Layout:
        image_pil = Image.fromarray(image)

        encoding = self._feature_extractor(image_pil, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**encoding)

        width, height = image_pil.size
        table_detection_results = (
            self._feature_extractor.post_process_object_detection(
                outputs,
                threshold=self._detection_threshold,
                target_sizes=[(height, width)],
            )[0]
        )

        table_block = self._post_process_results(table_detection_results)
        return table_block
