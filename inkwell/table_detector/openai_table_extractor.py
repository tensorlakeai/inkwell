import base64
import json
import os
from io import BytesIO

import numpy as np
import openai
from PIL import Image

from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.utils import TABLE_EXTRACTOR_PROMPT

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
OPENAI_MODEL = "gpt-4o-mini"


class OpenAITableExtractor(BaseTableExtractor):
    def __init__(self):
        self._load_client()

    def _load_client(self):
        self._client = openai.OpenAI(api_key=os.getenv(OPENAI_API_KEY_NAME))

    def _encode_image(self, image: np.ndarray) -> str:
        image_pil = Image.fromarray(image)
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    def process(self, image: np.ndarray) -> dict:
        img_str = self._encode_image(image)
        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_EXTRACTOR_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            },
                        }
                    ],
                },
            ],
        )
        response_content = response.choices[0].message.content
        response_dict = json.loads(response_content)
        return response_dict
