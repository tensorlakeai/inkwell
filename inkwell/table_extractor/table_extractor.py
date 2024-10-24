from enum import Enum


class TableExtractorType(Enum):
    TABLE_TRANSFORMER = "table_transformer"
    PHI3_VISION = "phi3_vision"
    OPENAI_GPT4O_MINI = "openai_gpt4o_mini"
    QWEN2_2B_VISION = "qwen2_2b_vision"
    PADDLE = "paddle"
    MINI_CPM = "minicpm"
