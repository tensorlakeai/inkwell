# pylint: disable=duplicate-code

import logging
from typing import List, Union

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.ocr.base import BaseOCR
from inkwell.ocr.config import _load_ocr_prompts
from inkwell.ocr.ocr import OCRType

_logger = logging.getLogger(__name__)


class Phi3VisionOCR(BaseOCR):
    def __init__(self):
        self._ocr_prompts = _load_ocr_prompts()
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            ModelType.PHI3_VISION_VLLM.value
        )

    @property
    def model_id(self) -> str:
        return OCRType.PHI3_VISION.value

    def process(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        system_prompt = self._ocr_prompts.system_prompt
        user_prompt = self._ocr_prompts.ocr_user_prompt
        return self._model_wrapper.process(image, user_prompt, system_prompt)
