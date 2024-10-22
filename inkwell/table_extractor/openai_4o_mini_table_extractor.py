import numpy as np

from inkwell.ocr.openai_4o_mini_ocr import OpenAI4OMiniOCR
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json


class OpenAI4OMiniTableExtractor(BaseTableExtractor):
    def __init__(self):
        self._load_client()
        self._prompt = _load_table_extractor_prompt()

    @property
    def model_id(self) -> str:
        return TableExtractorType.OPENAI_GPT4O_MINI.value

    def _load_client(self):
        self._client = OpenAI4OMiniOCR()

    @convert_markdown_to_json
    def process(self, image_batch: list[np.ndarray]) -> list[dict]:
        return self._client.process(
            image_batch,
            user_prompt=self._prompt.user_prompt,
            system_prompt=self._prompt.system_prompt,
        )
