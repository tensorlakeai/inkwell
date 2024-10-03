# flake8: noqa: E501

from inkwell.table_extractor.openai_4o_mini_table_extractor import (
    OpenAI4OMiniTableExtractor,
)
from inkwell.table_extractor.phi3v_table_extractor import Phi3VTableExtractor
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.table_extractor.table_transformer_extractor import (
    TableTransformerExtractor,
)
from inkwell.utils.env_utils import is_paddleocr_available, is_qwen2_available


class TableExtractorFactory:
    @staticmethod
    def get_table_extractor(
        table_extractor_type: TableExtractorType, **kwargs
    ):
        if table_extractor_type == TableExtractorType.TABLE_TRANSFORMER:
            return TableTransformerExtractor()
        if table_extractor_type == TableExtractorType.PHI3_VISION:
            return Phi3VTableExtractor(**kwargs)
        if table_extractor_type == TableExtractorType.OPENAI_GPT4O_MINI:
            return OpenAI4OMiniTableExtractor()
        if table_extractor_type == TableExtractorType.QWEN2_2B_VISION:
            if is_qwen2_available():
                from inkwell.table_extractor.qwen2_table_extractor import (  # pylint: disable=import-outside-toplevel,unused-import
                    Qwen2TableExtractor,
                )

                return Qwen2TableExtractor()
            raise ValueError(
                "Please install the latest transformers from source \
                        to use Qwen2 Vision OCR"
            )
        if table_extractor_type == TableExtractorType.PADDLE:
            if is_paddleocr_available():
                from inkwell.table_extractor.paddle_table_extractor import (  # pylint: disable=import-outside-toplevel,unused-import
                    PaddleTableExtractor,
                )

                return PaddleTableExtractor()
            raise ValueError("Please install paddleocr to use PaddleOCR")
        raise ValueError(
            f"Invalid table extractor type: {table_extractor_type}"
        )
