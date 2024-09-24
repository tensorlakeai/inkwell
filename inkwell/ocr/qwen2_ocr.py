import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType
from inkwell.ocr.utils import _load_ocr_config
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
)


class Qwen2VisionOCR(BaseOCR):
    def __init__(self, user_prompt: str = ""):
        self._cfg = _load_ocr_config()
        self._model_cfg = self._get_model_cfg()
        self._user_prompt = user_prompt

        self._load_model()
        self._load_processor()

    def _get_model_cfg(self):
        return self._cfg["ocr_models"][OCRType.QWEN2_VISION.value]

    def _load_model(self):
        if is_torch_cuda_available():
            _attn_implementation = (
                "flash_attention_2"
                if is_flash_attention_available()
                else "eager"
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_cfg["model_name_hf"],
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                _attn_implementation=_attn_implementation,
            )
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_cfg["model_name_hf"],
                torch_dtype="auto",
                _attn_implementation="eager",
            )

    def _load_processor(self):
        self._processor = AutoProcessor.from_pretrained(
            self._model_cfg["model_name_hf"]
        )

    def _get_system_prompt(self):
        return self._cfg["ocr_prompts"]["system_prompt"]

    def _get_user_prompt(self):
        return self._user_prompt

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

    def _preprocess_input(self, image: np.ndarray):
        image = Image.fromarray(image)

        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt()

        messages = self._get_messages(system_prompt, user_prompt)
        inputs = self._get_llm_input(messages, image)
        return inputs

    def _load_generation_args(self):
        return self._cfg["generation_args"]

    def process(self, image: np.ndarray) -> str:
        inputs = self._preprocess_input(image)
        generation_args = self._load_generation_args()

        if is_torch_cuda_available():
            inputs = inputs.to("cuda")

        generation_args = {
            "top_p": 0.95,
            "max_new_tokens": 2048,
        }
        output_ids = self._model.generate(**inputs, **generation_args)
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
