import json

import numpy as np

from inkwell.ocr.openai_ocr import OpenAI4OCR
from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.utils import TABLE_EXTRACTOR_PROMPT


class OpenAI4OTableExtractor(BaseTableExtractor):
    def __init__(self):
        self._load_client()

    def _load_client(self):
        self._client = OpenAI4OCR()

    def process(self, image: np.ndarray) -> dict:
        ocr_results = self._client.process(
            image, user_prompt=TABLE_EXTRACTOR_PROMPT
        )
        try:
            # try to decode as JSON
            ocr_results = ocr_results.replace("json", "").replace("```", "")
            ocr_results = json.loads(ocr_results)
        except json.JSONDecodeError:
            return ocr_results
        return ocr_results
