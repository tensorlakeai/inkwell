from enum import Enum


class TableExtractorType(Enum):
    TABLE_TRANSFORMER = "table_transformer"
    PHI3_VISION = "phi3_vision"
    OPENAI = "openai"
    QWEN2_2B_VISION = "qwen2_2b_vision"
    PADDLE = "paddle"
    OPENAI4O = "openai4o"
