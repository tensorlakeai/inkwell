# flake8: noqa: E501

from inkwell.table_detector.openai_table_extractor import OpenAITableExtractor
from inkwell.table_detector.phi3v_table_extractor import Phi3VTableExtractor
from inkwell.table_detector.table_detector import (
    TableDetectorType,
    TableExtractorType,
)
from inkwell.table_detector.table_transformer_detector import (
    TableTransformerDetector,
)
from inkwell.table_detector.table_transformer_extractor import (
    TableTransformerExtractor,
)
from inkwell.utils.env_utils import is_paddleocr_available, is_qwen2_available


class TableDetectorFactory:
    @staticmethod
    def get_table_detector(table_detector_type: TableDetectorType, **kwargs):
        if table_detector_type == TableDetectorType.TABLE_TRANSFORMER:
            return TableTransformerDetector(**kwargs)
        raise ValueError(f"Invalid table detector type: {table_detector_type}")


class TableExtractorFactory:
    @staticmethod
    def get_table_extractor(table_extractor_type: TableExtractorType):
        if table_extractor_type == TableExtractorType.TABLE_TRANSFORMER:
            return TableTransformerExtractor()
        if table_extractor_type == TableExtractorType.PHI3_VISION:
            return Phi3VTableExtractor()
        if table_extractor_type == TableExtractorType.OPENAI:
            return OpenAITableExtractor()
        if table_extractor_type == TableExtractorType.QWEN2_VISION:
            if is_qwen2_available():
                from inkwell.table_detector.qwen2_table_extractor import (  # pylint: disable=import-outside-toplevel,unused-import
                    Qwen2TableExtractor,
                )

                return Qwen2TableExtractor()
            raise ValueError(
                "Please install the latest transformers from source \
                        to use Qwen2 Vision OCR"
            )
        if table_extractor_type == TableExtractorType.PADDLE:
            if is_paddleocr_available():
                from inkwell.table_detector.paddle_table_extractor import (  # pylint: disable=import-outside-toplevel,unused-import
                    PaddleTableExtractor,
                )

                return PaddleTableExtractor()
            raise ValueError("Please install paddleocr to use PaddleOCR")
        raise ValueError(
            f"Invalid table extractor type: {table_extractor_type}"
        )
