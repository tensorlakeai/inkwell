# pylint: disable=duplicate-code

import logging

import numpy as np
import torch
from PIL import Image

from inkwell.models.base import BaseVisionModelWrapper
from inkwell.models.config import QWEN2_VISION_MODEL_CONFIG
from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
)

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Please install the latest transformers \
            from source to use Qwen2 models"
    ) from e

_logger = logging.getLogger(__name__)


class Qwen2VL2VModelWrapper(BaseVisionModelWrapper):
    def __init__(self):
        self._model_cfg = QWEN2_VISION_MODEL_CONFIG

        self._load_model()

    def _load_model(self):
        model_kwargs = {}
        if is_torch_cuda_available():
            _attn_implementation = (
                "flash_attention_2"
                if is_flash_attention_available()
                else "eager"
            )
            model_kwargs.update(
                {
                    "device_map": "cuda",
                    "torch_dtype": torch.bfloat16,
                    "_attn_implementation": _attn_implementation,
                }
            )
        else:
            model_kwargs.update(
                {
                    "_attn_implementation": "eager",
                    "torch_dtype": "auto",
                }
            )

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_cfg.model_name_hf, **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_cfg.model_name_hf,
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

    def _get_llm_input(self, messages, image: np.ndarray):
        text_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _preprocess_input(
        self, image: np.ndarray, user_prompt: str, system_prompt: str
    ):

        image = Image.fromarray(image)

        messages = self._get_messages(system_prompt, user_prompt)
        inputs = self._get_llm_input(messages, image)
        return inputs

    def _load_generation_args(self):
        return self._model_cfg.generation_args

    def process(
        self, image: np.ndarray, user_prompt: str, system_prompt: str
    ) -> str:
        inputs = self._preprocess_input(image, user_prompt, system_prompt)
        generation_args = self._load_generation_args()

        if is_torch_cuda_available():
            inputs = inputs.to("cuda")

        output_ids = self._model.generate(**inputs, **dict(generation_args))
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return output_text[0]


# Register the Qwen2 VL model loader
ModelRegistry.register_wrapper_loader(
    model_name=ModelType.QWEN2_2B_VISION.value, loader=Qwen2VL2VModelWrapper
)
