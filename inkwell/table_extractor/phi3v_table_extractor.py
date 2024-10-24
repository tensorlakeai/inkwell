# pylint: disable=duplicate-code

import logging
from typing import Optional

import numpy as np

from inkwell.ocr.phi3_ocr import Phi3VisionOCR
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class Phi3VTableExtractor(Phi3VisionOCR, BaseTableExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call the constructor of Phi3VisionOCR
        self._prompt = _load_table_extractor_prompt()

    @property
    def model_id(self) -> str:
        return TableExtractorType.PHI3_VISION.value

    @convert_markdown_to_json
    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        _logger.info("Running Phi3 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        return super().process(image_batch, user_prompt, system_prompt)
