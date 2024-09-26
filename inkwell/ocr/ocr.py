from enum import Enum


class OCRType(Enum):
    TESSERACT = "tesseract"
    PHI3_VISION = "phi3_vision"
    QWEN2_VISION = "qwen2_vision"
    PADDLE = "paddle"
