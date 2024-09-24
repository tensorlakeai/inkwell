# pylint: disable=duplicate-code

import json

import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType
from inkwell.ocr.utils import _load_ocr_config
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
)

IMAGE_PLACEHOLDER = "<|image_{i}|>\n"
NUM_IMAGES = 1  # Change this later if we want to support multiple images


class Phi3VisionOCR(BaseOCR):
    def __init__(self, user_prompt: str = ""):
        self._cfg = _load_ocr_config()
        self._model_cfg = self._get_model_cfg()

        self._user_prompt = user_prompt
        self._load_model()
        self._load_processor()

    def _get_model_cfg(self):
        return self._cfg["ocr_models"][OCRType.PHI3_VISION.value]

    def _load_model(self):
        if is_torch_cuda_available():
            _attn_implementation = (
                "flash_attention_2"
                if is_flash_attention_available()
                else "eager"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_cfg["model_name_hf"],
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation=_attn_implementation,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_cfg["model_name_hf"],
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="eager",
            )

    def _load_processor(self):
        self._processor = AutoProcessor.from_pretrained(
            self._model_cfg["model_name_hf"],
            trust_remote_code=True,
            num_crops=4,
        )

    def _get_system_prompt(self):
        return self._cfg["ocr_prompts"]["system_prompt"]

    def _get_user_prompt(self):
        if not self._user_prompt:
            return self._cfg["ocr_prompts"]["ocr_user_prompt"]
        return self._user_prompt

    def _preprocess_image_placeholder(self, num_images: int):
        return "".join(
            [IMAGE_PLACEHOLDER.format(i=i + 1) for i in range(num_images)]
        )

    @staticmethod
    def _get_messages(
        system_prompt: str, image_placeholder: str, user_prompt: str
    ):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_placeholder + user_prompt},
        ]

    def _get_llm_input(self, messages, image: np.ndarray):
        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, image, return_tensors="pt")

        return inputs

    def _preprocess_input(self, image: np.ndarray):
        image = Image.fromarray(image)

        system_prompt = self._get_system_prompt()
        image_placeholder = self._preprocess_image_placeholder(NUM_IMAGES)
        user_prompt = self._get_user_prompt()

        messages = self._get_messages(
            system_prompt, image_placeholder, user_prompt
        )
        inputs = self._get_llm_input(messages, image)
        return inputs

    def _postprocess_output(self, output: str) -> dict:
        return json.loads(output.replace("json\n", "").replace("```", ""))

    def _load_generation_args(self):
        return self._model_cfg["generation_args"]

    def process(self, image: np.ndarray) -> str:
        inputs = self._preprocess_input(image)
        generation_args = self._load_generation_args()

        if is_torch_cuda_available():
            inputs = inputs.to("cuda")

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response
