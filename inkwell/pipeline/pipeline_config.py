from typing import Any, Dict, Union

from pydantic import BaseModel

from inkwell.layout_detector import LayoutDetectorType
from inkwell.models import InferenceBackend
from inkwell.ocr import OCRType
from inkwell.table_detector import TableDetectorType
from inkwell.table_extractor import TableExtractorType


class PipelineConfig(BaseModel):
    layout_detector: Union[LayoutDetectorType, None] = None
    ocr_detector: Union[OCRType, None] = None
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = None
    inference_backend: Union[InferenceBackend, None] = None


class DefaultPipelineConfig(PipelineConfig):
    table_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.VLLM

    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.TESSERACT
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.TABLE_TRANSFORMER
    )


class DefaultGPUPipelineConfig(PipelineConfig):
    table_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.VLLM
    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.PHI3_VISION
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.PHI3_VISION
    )
