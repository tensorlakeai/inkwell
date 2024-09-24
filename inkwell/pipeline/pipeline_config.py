from typing import Any, Dict, Union

from pydantic import BaseModel

from inkwell.layout_detector import LayoutDetectorType
from inkwell.ocr import OCRType
from inkwell.table_detector import TableDetectorType, TableExtractorType


class PipelineConfig(BaseModel):
    layout_detector: Union[LayoutDetectorType, None] = None
    ocr_detector: Union[OCRType, None] = None
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = None


class DefaultGPUPipelineConfig(PipelineConfig):
    table_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}

    layout_detector: LayoutDetectorType = LayoutDetectorType.FASTER_RCNN
    ocr_detector: OCRType = OCRType.PHI3_VISION
    table_detector: TableDetectorType = None
    table_extractor: TableExtractorType = TableExtractorType.PHI3_VISION


class DefaultCPUPipelineConfig(PipelineConfig):
    table_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}

    layout_detector: LayoutDetectorType = LayoutDetectorType.FASTER_RCNN
    ocr_detector: OCRType = OCRType.TESSERACT
    table_detector: TableDetectorType = None
    table_extractor: TableExtractorType = TableExtractorType.TABLE_TRANSFORMER
