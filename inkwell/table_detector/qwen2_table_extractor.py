import json
import logging

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.utils import (
    TABLE_EXTRACTOR_PROMPT,
    _load_table_detector_config,
)

_logger = logging.getLogger(__name__)


class Qwen2TableExtractor(BaseTableExtractor):
    def __init__(self):
        self._cfg = _load_table_detector_config()
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            ModelType.QWEN2_2B_VISION.value
        )

    def process(self, image: np.ndarray) -> dict:
        _logger.info("Running Qwen2 Vision Table Extractor")
        system_prompt = self._cfg["table_extraction_prompts"]["system_prompt"]
        result = self._model_wrapper.process(
            image, TABLE_EXTRACTOR_PROMPT, system_prompt
        )
        formatted_result = result.replace("```json", "").replace("```", "")
        result_dict = json.loads(formatted_result)

        return result_dict
