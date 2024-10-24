from typing import Any, Dict, Union

from pydantic import BaseModel

from inkwell.figure_extractor import FigureExtractorType
from inkwell.layout_detector import LayoutDetectorType
from inkwell.models import InferenceBackend
from inkwell.ocr import OCRType
from inkwell.reading_order import ReadingOrderDetectorType
from inkwell.table_detector import TableDetectorType
from inkwell.table_extractor import TableExtractorType


class PipelineConfig(BaseModel):
    layout_detector: Union[LayoutDetectorType, None] = None
    ocr_detector: Union[OCRType, None] = None
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = None
    inference_backend: Union[InferenceBackend, None] = None
    reading_order_detector: Union[ReadingOrderDetectorType, None] = None


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
    figure_extractor: Union[FigureExtractorType, None] = None


class DefaultGPUPipelineConfig(PipelineConfig):
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
    figure_extractor: Union[FigureExtractorType, None] = None


class OpenAIGPT4oMiniPipelineConfig(PipelineConfig):
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.VLLM
    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.OPENAI_GPT4O_MINI
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.OPENAI_GPT4O_MINI
    )
    figure_extractor: Union[FigureExtractorType, None] = (
        FigureExtractorType.OPENAI_GPT4O_MINI
    )


class Phi3VisionPipelineConfig(PipelineConfig):
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.HUGGINGFACE
    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.PHI3_VISION
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.PHI3_VISION
    )
    figure_extractor: Union[FigureExtractorType, None] = (
        FigureExtractorType.PHI3_VISION
    )


class Qwen2VisionPipelineConfig(PipelineConfig):
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.VLLM
    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.QWEN2_2B_VISION
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.QWEN2_2B_VISION
    )
    figure_extractor: Union[FigureExtractorType, None] = None


class MiniCPMPipelineConfig(PipelineConfig):
    layout_detector_kwargs: Dict[str, Any] = {"detection_threshold": 0.4}
    inference_backend: InferenceBackend = InferenceBackend.VLLM
    layout_detector: Union[LayoutDetectorType, None] = (
        LayoutDetectorType.FASTER_RCNN
    )
    ocr_detector: Union[OCRType, None] = OCRType.MINI_CPM
    table_detector: Union[TableDetectorType, None] = None
    table_extractor: Union[TableExtractorType, None] = (
        TableExtractorType.MINI_CPM
    )
    figure_extractor: Union[FigureExtractorType, None] = (
        FigureExtractorType.MINI_CPM
    )
