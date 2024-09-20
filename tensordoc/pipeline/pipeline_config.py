from typing import Any, Dict

from pydantic import BaseModel

from tensordoc.layout_detector import LayoutDetectorType
from tensordoc.ocr import OCRType
from tensordoc.table_detector import TableDetectorType, TableExtractorType


class PipelineConfig(BaseModel):
    table_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}

    layout_detector: LayoutDetectorType = LayoutDetectorType.FASTER_RCNN
    ocr_detector: OCRType = OCRType.TESSERACT
    table_detector: TableDetectorType = None
    table_extractor: TableExtractorType = TableExtractorType.TABLE_TRANSFORMER
