import logging

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class Qwen2TableExtractor(BaseTableExtractor):
    def __init__(self):
        self._prompt = _load_table_extractor_prompt()
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            ModelType.QWEN2_2B_VISION.value
        )

    @property
    def model_id(self) -> str:
        return TableExtractorType.QWEN2_2B_VISION.value

    @convert_markdown_to_json
    def process(self, image: np.ndarray) -> dict:
        _logger.info("Running Qwen2 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        result = self._model_wrapper.process(image, user_prompt, system_prompt)

        return result.strip()
