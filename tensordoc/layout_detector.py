from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import layoutparser as lp
import numpy as np

DETECTRON2_FASTER_RCNN_R_50_FPN_3X_CONFIG = (
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
)

FASTER_RCNN_LABEL_MAP = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}


class LayoutDetectorType(Enum):
    DETECTRON2 = "detectron2"


class LayoutDetector(ABC):
    """
    Abstract class for layout detection.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> List[lp.elements.Layout]: ...


class Detectron2LayoutDetector(LayoutDetector):
    """
    Layout detector using Detectron2 and LayoutParser.
    """

    def __init__(self):
        self.model = lp.models.Detectron2LayoutModel(
            config_path=DETECTRON2_FASTER_RCNN_R_50_FPN_3X_CONFIG,
            label_map=FASTER_RCNN_LABEL_MAP,
        )

    def process(self, image: np.ndarray) -> List[lp.elements.Layout]:
        return self.model.detect(image)


class LayoutDetectorFactory:
    """
    Factory class for layout detectors.
    """

    @staticmethod
    def get_layout_detector(
        layout_detector_type: LayoutDetectorType,
    ) -> LayoutDetector:
        """
        Get a layout detector based on the type.
        """
        if layout_detector_type == LayoutDetectorType.DETECTRON2:
            return Detectron2LayoutDetector()
        raise ValueError(
            f"Invalid layout detector type: {layout_detector_type}"
        )
