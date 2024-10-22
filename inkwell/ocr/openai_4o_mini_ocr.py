import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
import openai
from PIL import Image

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.config import OPENAI_OCR_MODEL_CONFIG, _load_ocr_prompts
from inkwell.ocr.ocr import OCRType

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"


class OpenAI4OMiniOCR(BaseOCR):
    def __init__(self):
        self._default_prompts = _load_ocr_prompts()
        self._model_cfg = OPENAI_OCR_MODEL_CONFIG

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

    def _call_client(
        self, system_prompt: str, user_prompt: str, img: np.ndarray
    ) -> str:
        messages = self._create_message(system_prompt, user_prompt, img)
        response = self._client.chat.completions.create(
            model=self._model_cfg.model_name_openai, messages=messages
        )
        return response.choices[0].message.content

    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        """
        Processes the image(s) and returns the text(s) detected.

        Args:
            image (np.ndarray or list[np.ndarray]): The image(s) to process.
            user_prompt (str, optional): The user prompt. Defaults to None.
            system_prompt (str, optional): The system prompt. Defaults to None.

        Returns:
            str or list[str]: The text(s) detected.
        """
        if not user_prompt:
            user_prompt = self._default_prompts.ocr_user_prompt

        if not system_prompt:
            system_prompt = self._default_prompts.system_prompt

        results = []
        for img in image_batch:
            response_content = self._call_client(
                system_prompt, user_prompt, img
            )
            results.append(response_content)
        return results
