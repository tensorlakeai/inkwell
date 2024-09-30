from enum import Enum


class TableDetectorType(Enum):
    TABLE_TRANSFORMER = "table_transformer_detector"


class TableExtractorType(Enum):
    TABLE_TRANSFORMER = "table_transformer_extractor"
    PHI3_VISION = "phi3_vision"
    OPENAI = "openai_table_extractor"
    QWEN2_2B_VISION = "qwen2_2b_vision"
    PADDLE = "paddle_table_extractor"
    OPENAI4O = "openai_table_extractor"
