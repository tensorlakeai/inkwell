# flake8: noqa: E501

from inkwell.layout_detector.base import BaseLayoutDetector
from inkwell.layout_detector.layout_detector import LayoutDetectorType
from inkwell.utils.env_utils import is_paddleocr_available


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
        from inkwell.layout_detector.faster_rcnn_detector import (
            FasterRCNNLayoutDetector,
        )
        from inkwell.layout_detector.layoutlmv3_detector import LayoutLMv3Detector
        if layout_detector_type == LayoutDetectorType.FASTER_RCNN:
            return FasterRCNNLayoutDetector(**kwargs)
        if layout_detector_type == LayoutDetectorType.LAYOUTLMV3:
            return LayoutLMv3Detector(**kwargs)
        if layout_detector_type == LayoutDetectorType.PADDLE:
            if is_paddleocr_available():
                from inkwell.layout_detector.paddle_detector import (  # pylint: disable=import-outside-toplevel
                    PaddleDetector,
                )

                return PaddleDetector(**kwargs)
        raise ValueError(
            f"Invalid layout detector type: {layout_detector_type}"
        )
