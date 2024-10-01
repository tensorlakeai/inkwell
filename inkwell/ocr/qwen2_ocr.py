import logging

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.ocr.utils import _load_ocr_config

_logger = logging.getLogger(__name__)


class Qwen2OCR:
    def __init__(self):
        self._cfg = _load_ocr_config()
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            ModelType.QWEN2_2B_VISION.value
        )

    def process(self, image: np.ndarray) -> str:
        _logger.info("Running Qwen2 OCR")
        system_prompt = self._cfg["ocr_prompts"]["system_prompt"]
        user_prompt = self._cfg["ocr_prompts"]["ocr_user_prompt"]
        return self._model_wrapper.process(image, user_prompt, system_prompt)
