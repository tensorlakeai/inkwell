import logging

import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from inkwell.models.base import BaseVisionModelWrapper
from inkwell.models.config import PHI3_VISION_MODEL_CONFIG
from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
)

_logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<|image_{i}|>\n"
NUM_IMAGES = 1  # Change this later if we want to support multiple images


class Phi3VisionModelWrapper(BaseVisionModelWrapper):
    def __init__(self, **kwargs):
        self._model_cfg = PHI3_VISION_MODEL_CONFIG
        self._model_path = kwargs.get("model_path", None)
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
                    "_attn_implementation": _attn_implementation,
                }
            )
        else:
            model_kwargs["_attn_implementation"] = "eager"
        model_kwargs.update(
            {
                "trust_remote_code": True,
                "torch_dtype": "auto",
            }
        )
        model_path = self._model_path or self._model_cfg.model_name_hf
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=4,
        )

    def _preprocess_image_placeholder(self, num_images: int):
        return "".join(
            [IMAGE_PLACEHOLDER.format(i=i + 1) for i in range(num_images)]
        )

    def _get_llm_input(self, messages, images: list[Image.Image]):
        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = [
            self._processor(prompt, image, return_tensors="pt")
            for image in images
        ]
        return inputs

    def _get_messages(
        self, system_prompt: str, image_placeholder: str, user_prompt: str
    ):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_placeholder + user_prompt},
        ]

    def _preprocess_input(
        self,
        image_batch: list[np.ndarray],
        system_prompt: str = None,
        user_prompt: str = None,
    ):
        images = [Image.fromarray(image) for image in image_batch]
        image_placeholder = self._preprocess_image_placeholder(NUM_IMAGES)
        messages = self._get_messages(
            system_prompt, image_placeholder, user_prompt
        )
        inputs = self._get_llm_input(messages, images)
        return inputs

    def _load_generation_args(self):
        return PHI3_VISION_MODEL_CONFIG.generation_args

    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: str,
        system_prompt: str,
    ) -> list[str]:
        inputs = self._preprocess_input(
            image_batch, system_prompt, user_prompt
        )
        generation_args = self._load_generation_args()

        if is_torch_cuda_available():
            inputs = inputs.to("cuda")

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **dict(generation_args),
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


# Register the Phi3 model loader
ModelRegistry.register_wrapper_loader(
    model_name=ModelType.PHI3_VISION_HF.value, loader=Phi3VisionModelWrapper
)
