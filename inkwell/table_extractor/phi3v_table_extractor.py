# pylint: disable=duplicate-code

import json
import logging

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType

_logger = logging.getLogger(__name__)


class Phi3VTableExtractor(BaseTableExtractor):
    def __init__(self):
        self._prompt = _load_table_extractor_prompt()
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            ModelType.PHI3_VISION.value
        )

    @property
    def model_id(self) -> str:
        return TableExtractorType.PHI3_VISION.value

    def process(self, image: np.ndarray) -> dict:
        _logger.info("Running Phi3 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        result = self._model_wrapper.process(image, user_prompt, system_prompt)
        formatted_result = result.replace("```json", "").replace("```", "")
        result_dict = json.loads(formatted_result)

        return result_dict
