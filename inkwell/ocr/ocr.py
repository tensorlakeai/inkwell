from enum import Enum


class OCRType(Enum):
    TESSERACT = "tesseract"
    PHI3_VISION = "phi3_vision"
    QWEN2_2B_VISION = "qwen2_2b_vision"
    PADDLE = "paddle"
    OPENAI_GPT4O_MINI = "openai_gpt4o_mini"
    MINI_CPM = "minicpm"
