from enum import Enum


class TableDetectorType(Enum):
    TABLE_TRANSFORMER = "table_transformer_detector"


class TableExtractorType(Enum):
    TABLE_TRANSFORMER = "table_transformer_extractor"
    PHI3_VISION = "phi3v_table_extractor"
    OPENAI = "openai_table_extractor"
    QWEN2_VISION = "qwen2_table_extractor"
    PADDLE = "paddle_table_extractor"
