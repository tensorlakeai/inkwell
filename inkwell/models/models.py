from enum import Enum


class ModelType(Enum):
    PHI3_VISION_HF = "phi3_vision_hf"
    PHI3_VISION_VLLM = "phi3_vision_vllm"
    MINI_CPM_VLLM = "minicpm_vllm"
    QWEN2_2B_VISION_HF = "qwen2_2b_vision_hf"
    QWEN2_2B_VISION_VLLM = "qwen2_2b_vision_vllm"


class InferenceBackend(Enum):
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
