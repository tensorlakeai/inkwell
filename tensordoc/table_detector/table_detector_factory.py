from tensordoc.table_detector.table_detector import (
    TableDetectorType,
    TableSegmentationType,
)
from tensordoc.table_detector.table_transformer_detector import (
    TableTransformerDetector,
)
from tensordoc.table_detector.table_transformer_segmentation import (
    TableTransformerSegmentation,
)


class TableDetectorFactory:
    @staticmethod
    def get_table_detector(table_detector_type: TableDetectorType):
        if table_detector_type == TableDetectorType.TABLE_TRANSFORMER:
            return TableTransformerDetector()
        raise ValueError(f"Invalid table detector type: {table_detector_type}")


class TableSegmentationFactory:
    @staticmethod
    def get_table_segmentation(table_segmentation_type: TableSegmentationType):
        if table_segmentation_type == TableSegmentationType.TABLE_TRANSFORMER:
            return TableTransformerSegmentation()
        raise ValueError(
            f"Invalid table segmentation type: {table_segmentation_type}"
        )
