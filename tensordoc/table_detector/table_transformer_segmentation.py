from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from tensordoc.components import Layout, Rectangle, TextBlock
from tensordoc.table_detector.base import BaseTableSegmentation
from tensordoc.table_detector.table_detector import TableSegmentationType
from tensordoc.table_detector.utils import load_table_detector_config


class TableTransformerSegmentation(BaseTableSegmentation):
    def __init__(self):
        self._feature_extractor = DetrImageProcessor()
        self._cfg = load_table_detector_config(
            TableSegmentationType.TABLE_TRANSFORMER
        )
        self._load_processor()
        self._load_model()

    def _load_processor(self):
        self._processor = DetrImageProcessor()

    def _load_model(self):
        self._model = TableTransformerForObjectDetection.from_pretrained(
            self._cfg["model_name_hf"]
        )

    def _post_process_results(self, outputs: Dict) -> Layout:
        blocks = []
        scores = outputs["scores"]
        labels = outputs["labels"]
        boxes = outputs["boxes"]

        total = len(scores)
        for i in range(total):

            block = TextBlock(
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
            blocks=[b for b in blocks if b.type != "Table"]
        )

        return segmented_table_layout

    def process(self, image: np.ndarray) -> Layout:

        image = Image.fromarray(image)
        encoding = self._processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**encoding)

        target_sizes = [image.size[::-1]]
        results = self._processor.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes
        )[0]
        table_blocks = self._post_process_results(results)

        return table_blocks
