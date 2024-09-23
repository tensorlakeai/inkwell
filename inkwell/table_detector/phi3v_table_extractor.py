import json

import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.table_detector import TableExtractorType
from inkwell.table_detector.utils import (
    TABLE_EXTRACTOR_PROMPT,
    load_table_detector_config,
)
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
)


class Phi3VTableExtractor(BaseTableExtractor):
    def __init__(self):

        self._cfg = load_table_detector_config(TableExtractorType.PHI3V)
        self._load_model()
        self._load_processor()

    def _load_model(self):
        _attn_implementation = (
            "flash_attention_2" if is_flash_attention_available() else "eager"
        )

        if is_torch_cuda_available():
            self._model = AutoModelForCausalLM.from_pretrained(
                self._cfg["model_name_hf"],
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation=_attn_implementation,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._cfg["model_name_hf"],
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation=_attn_implementation,
            )

    def _load_processor(self):
        self._processor = AutoProcessor.from_pretrained(
            self._cfg["model_name_hf"], trust_remote_code=True, num_crops=4
        )

    def _preprocess_image(self, image: np.ndarray):
        image = Image.fromarray(image)
        image_placeholder = f"<|image_{1}|>\n"
        prompt = TABLE_EXTRACTOR_PROMPT

        messages = [{"role": "user", "content": prompt + image_placeholder}]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, image, return_tensors="pt")
        if is_torch_cuda_available():
            inputs = inputs.to("cuda")
        return inputs

    def process(self, image: np.ndarray) -> dict:
        inputs = self._preprocess_image(image)
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        response_json = json.loads(
            response.replace("json\n", "").replace("```", "")
        )
        return response_json
