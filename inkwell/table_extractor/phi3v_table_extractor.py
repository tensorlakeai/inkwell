# pylint: disable=duplicate-code

import logging

import numpy as np

from inkwell.ocr.phi3_ocr import Phi3VisionOCR
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class Phi3VTableExtractor(BaseTableExtractor):
    def __init__(self, **kwargs):
        self._prompt = _load_table_extractor_prompt()
        self._ocr_client = Phi3VisionOCR(**kwargs)

    @property
    def model_id(self) -> str:
        return TableExtractorType.PHI3_VISION.value

    @convert_markdown_to_json
    def process(self, image: np.ndarray) -> dict:
        _logger.info("Running Phi3 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        result = self._ocr_client.process(image, user_prompt, system_prompt)

        return result
