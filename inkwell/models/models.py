from enum import Enum


class ModelType(Enum):
    PHI3_VISION = "phi3_vision"
    PHI3_VISION_VLLM = "phi3_vision_vllm"
    QWEN2_2B_VISION = "qwen2_2b_vision"
    QWEN2_2B_VISION_VLLM = "qwen2_2b_vision_vllm"


class InferenceBackend(Enum):
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
