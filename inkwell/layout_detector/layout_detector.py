from enum import Enum


class LayoutDetectorType(Enum):
    FASTER_RCNN = "faster_rcnn"
    LAYOUTLMV3 = "layoutlmv3"
    PADDLE = "paddle"
