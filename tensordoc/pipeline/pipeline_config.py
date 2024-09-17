from typing import Any, Dict

from pydantic import BaseModel

from tensordoc.layout_detector import LayoutDetectorType
from tensordoc.ocr import OCRType
from tensordoc.table_detector import TableDetectorType, TableSegmentationType


class PipelineConfig(BaseModel):
    layout_detector: LayoutDetectorType = LayoutDetectorType.FASTER_RCNN
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.5}
    ocr_detector: OCRType = OCRType.TESSERACT
    table_detector: TableDetectorType = TableDetectorType.TABLE_TRANSFORMER
    table_segmentation_detector: TableSegmentationType = (
        TableSegmentationType.TABLE_TRANSFORMER
    )
