import base64
import os
from io import BytesIO

import numpy as np
import openai
from PIL import Image

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType
from inkwell.ocr.utils import _load_ocr_config

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"


class OpenAI4OCR(BaseOCR):
    def __init__(self):
        self._cfg = _load_ocr_config()
        self._model_cfg = self._cfg["ocr_models"][
            OCRType.OPENAI_GPT4O_MINI.value
        ]
        self._load_client()

    @property
    def model_id(self) -> str:
        return OCRType.OPENAI_GPT4O_MINI.value

    def _load_client(self):
        self._client = openai.OpenAI(api_key=os.getenv(OPENAI_API_KEY_NAME))

    def _encode_image(self, image: np.ndarray) -> str:
        image_pil = Image.fromarray(image)
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    def _create_message(
        self, system_prompt: str, user_prompt: str, image: np.ndarray
    ) -> str:
        img_str = self._encode_image(image)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        },
                    },
                ],
            },
        ]

    def process(self, image: np.ndarray, user_prompt: str = None) -> str:
        if not user_prompt:
            user_prompt = self._cfg["ocr_prompts"]["ocr_user_prompt"]

        messages = self._create_message(
            self._cfg["ocr_prompts"]["system_prompt"], user_prompt, image
        )

        response = self._client.chat.completions.create(
            model=self._model_cfg["model_name_openai"], messages=messages
        )
        response_content = response.choices[0].message.content
        return response_content
