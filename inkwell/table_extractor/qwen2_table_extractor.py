import logging

import numpy as np

from inkwell.ocr.qwen2_ocr import Qwen2OCR
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class Qwen2TableExtractor(BaseTableExtractor):
    def __init__(self, **kwargs):
        self._prompt = _load_table_extractor_prompt()
        self._ocr_client = Qwen2OCR(**kwargs)

    @property
    def model_id(self) -> str:
        return TableExtractorType.QWEN2_2B_VISION.value

    @convert_markdown_to_json
    def process(self, image_batch: list[np.ndarray]) -> list[dict]:
        _logger.info("Running Qwen2 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        return self._ocr_client.process(
            image_batch, user_prompt, system_prompt
        )
