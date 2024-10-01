from inkwell.table_detector.table_detector import TableDetectorType
from inkwell.table_detector.table_transformer_detector import (
    TableTransformerDetector,
)


class TableDetectorFactory:
    @staticmethod
    def get_table_detector(table_detector_type: TableDetectorType, **kwargs):
        if table_detector_type == TableDetectorType.TABLE_TRANSFORMER:
            return TableTransformerDetector(**kwargs)
        raise ValueError(f"Invalid table detector type: {table_detector_type}")
