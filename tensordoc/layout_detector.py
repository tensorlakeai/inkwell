from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import layoutparser as lp
import numpy as np

from tensordoc.config.utils import load_layout_detector_config


class LayoutDetectorType(Enum):
    FASTER_RCNN = "faster_rcnn"


class LayoutDetector(ABC):
    """
    Abstract class for layout detection.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> List[lp.elements.Layout]: ...


class FasterRCNNLayoutDetector(LayoutDetector):
    """
    Faster RCNN based layout detector using Detectron2.
    """

    def __init__(self, **kwargs):

        self._config = load_layout_detector_config()["DETECTRON2_FASTER_RCNN"]
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        if "model_path" not in kwargs:
            model_path = self._config["WEIGHTS"]
        else:
            model_path = kwargs["model_path"]

        self._model = lp.models.Detectron2LayoutModel(
            model_path=model_path,
            config_path=self._config["CONFIG"],
            label_map=self._config["LABEL_MAP"],
            **kwargs,
        )

    def process(self, image: np.ndarray) -> List[lp.elements.Layout]:
        return self._model.detect(image)


class LayoutDetectorFactory:
    """
    Factory class for layout detectors.
    """

    @staticmethod
    def get_layout_detector(
        layout_detector_type: LayoutDetectorType,
        **kwargs,
    ) -> LayoutDetector:
        """
        Get a layout detector based on the type of layout detection engine.
        """
        if layout_detector_type == LayoutDetectorType.FASTER_RCNN:
            return FasterRCNNLayoutDetector(**kwargs)
        raise ValueError(
            f"Invalid layout detector type: {layout_detector_type}"
        )
