# pylint: disable=import-outside-toplevel
# flake8: noqa: F401

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import InferenceBackend
from inkwell.models.phi3_vision_model_loader import Phi3VisionModelWrapper
from inkwell.models.qwen2_2b_vision_model_loader import Qwen2VL2VModelWrapper
from inkwell.utils.env_utils import is_vllm_available

__all__ = [
    "Phi3VisionModelWrapper",
    "Qwen2VL2VModelWrapper",
    "ModelRegistry",
    "InferenceBackend",
]

if is_vllm_available():
    from inkwell.models.minicpm_model_loader_vllm import (
        MiniCPMModelWrapperVLLM,
    )
    from inkwell.models.phi3_vision_model_loader_vllm import (
        Phi3VisionModelWrapperVLLM,
    )
    from inkwell.models.qwen2_2b_vision_model_loader_vllm import (
        Qwen2VL2VModelWrapperVLLM,
    )

    __all__.extend(
        [
            "Phi3VisionModelWrapperVLLM",
            "Qwen2VL2VModelWrapperVLLM",
            "MiniCPMModelWrapperVLLM",
        ]
    )
