from inkwell.layout_detector.base import BaseLayoutDetector, LayoutDetectorType
from inkwell.layout_detector.faster_rcnn_detector import (
    FasterRCNNLayoutDetector,
)
from inkwell.layout_detector.layoutlmv3_detector import LayoutLMv3Detector


class LayoutDetectorFactory:
    """
    Factory class for layout detectors.
    """

    @staticmethod
    def get_layout_detector(
        layout_detector_type: LayoutDetectorType,
        **kwargs,
    ) -> BaseLayoutDetector:
        """
        Get a layout detector based on the type of layout detection engine.
        """
        if layout_detector_type == LayoutDetectorType.FASTER_RCNN:
            return FasterRCNNLayoutDetector(**kwargs)
        if layout_detector_type == LayoutDetectorType.LAYOUTLMV3:
            return LayoutLMv3Detector(**kwargs)
        raise ValueError(
            f"Invalid layout detector type: {layout_detector_type}"
        )
