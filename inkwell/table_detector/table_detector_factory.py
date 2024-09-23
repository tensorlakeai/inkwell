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
        if table_extractor_type == TableExtractorType.PHI3V:
            return Phi3VTableExtractor()
        if table_extractor_type == TableExtractorType.OPENAI:
            return OpenAITableExtractor()
        raise ValueError(
            f"Invalid table extractor type: {table_extractor_type}"
        )
