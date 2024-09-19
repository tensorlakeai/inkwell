from tensordoc.table_detector.table_detector import (
    TableDetectorType,
    TableExtractorType,
)
from tensordoc.table_detector.table_transformer_detector import (
    TableTransformerDetector,
)
from tensordoc.table_detector.table_transformer_extractor import (
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
        raise ValueError(
            f"Invalid table extractor type: {table_extractor_type}"
        )
