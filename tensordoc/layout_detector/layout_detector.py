from tensordoc.layout_detector.base import (
    BaseLayoutDetector,
    LayoutDetectorType,
)
from tensordoc.layout_detector.dit_detector import DitLayoutDetector
from tensordoc.layout_detector.faster_rcnn_detector import (
    FasterRCNNLayoutDetector,
)


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
        if layout_detector_type == LayoutDetectorType.DIT:
            return DitLayoutDetector(**kwargs)
        raise ValueError(
            f"Invalid layout detector type: {layout_detector_type}"
        )
