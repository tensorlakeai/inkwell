from inkwell.reading_order.noop_reading_order_detector import (
    NoOpReadingOrderDetector,
)
from inkwell.reading_order.reading_order_detector import (
    ReadingOrderDetectorType,
)


class ReadingOrderDetectorFactory:
    @staticmethod
    def get_reading_order_detector(
        reading_order_detector_type: ReadingOrderDetectorType,
    ):
        if (
            reading_order_detector_type
            == ReadingOrderDetectorType.NO_OP_DETECTOR
        ):
            return NoOpReadingOrderDetector()

        raise ValueError(
            f"Invalid reading order detector type: {reading_order_detector_type}"
        )
