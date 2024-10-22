# pylint: disable=duplicate-code

import logging

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams

from inkwell.models.base import BaseVisionModelWrapper
from inkwell.models.config import QWEN2_VISION_MODEL_CONFIG
from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
    is_vllm_available,
)

_logger = logging.getLogger(__name__)


class Qwen2VL2VModelWrapperVLLM(BaseVisionModelWrapper):
    def __init__(self, **kwargs):
        self._model_cfg = QWEN2_VISION_MODEL_CONFIG
        self._model_path = kwargs.get("model_path", None)
        self._load_model()

    def _load_model(self):
        if not (
            is_torch_cuda_available()
            and is_flash_attention_available()
            and is_vllm_available()
        ):
            raise ValueError(
                "vLLM based models work best with \
            flash-attention and modern GPUs"
            )

        model_path = self._model_path or self._model_cfg.model_name_hf
        _logger.info("Loading model from path: %s", model_path)
        self._model = LLM(
            model=model_path,
            quantization="fp8",
            dtype="bfloat16",
        )

    @staticmethod
    def _get_messages(system_prompt: str, user_prompt: str):
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def _preprocess_input(self, system_prompt: str, user_prompt: str) -> str:

        messages = self._get_messages(system_prompt, user_prompt)
        text_prompt = self._model.get_tokenizer().apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return text_prompt

    def _load_generation_args(self):
        generation_args = dict(QWEN2_VISION_MODEL_CONFIG.generation_args)
        if "max_new_tokens" in generation_args:
            generation_args["max_tokens"] = generation_args.pop(
                "max_new_tokens"
            )
        return generation_args

    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: str,
        system_prompt: str,
    ) -> list[str]:
        text_prompt = self._preprocess_input(system_prompt, user_prompt)
        generation_args = self._load_generation_args()
        sampling_params = SamplingParams(**generation_args)

        data = [
            {
                "prompt": text_prompt,
                "multi_modal_data": {"image": Image.fromarray(img)},
            }
            for img in image_batch
        ]
        outputs = self._model.generate(data, sampling_params=sampling_params)

        return [output.outputs[0].text for output in outputs]


# Register the Qwen2 2B VL VLLM model loader
ModelRegistry.register_wrapper_loader(
    model_name=ModelType.QWEN2_2B_VISION_VLLM.value,
    loader=Qwen2VL2VModelWrapperVLLM,
)
