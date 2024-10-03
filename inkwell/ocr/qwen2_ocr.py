# pylint: disable=duplicate-code

import logging
from typing import List, Union

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import InferenceBackend, ModelType
from inkwell.ocr.base import BaseOCR
from inkwell.ocr.config import _load_ocr_prompts
from inkwell.ocr.ocr import OCRType

_logger = logging.getLogger(__name__)


class Qwen2OCR(BaseOCR):
    def __init__(self, **kwargs):
        self._ocr_prompts = _load_ocr_prompts()
        self._inference_backend = kwargs.get(
            "inference_backend", InferenceBackend.VLLM
        )
        self._model_wrapper = (
            ModelType.PHI3_VISION_VLLM.value
            if self._inference_backend == InferenceBackend.VLLM
            else ModelType.PHI3_VISION.value
        )
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            self._model_wrapper
        )

    @property
    def model_id(self) -> str:
        return OCRType.QWEN2_2B_VISION.value

    def process(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        _logger.info("Running Qwen2 OCR")

        system_prompt = self._ocr_prompts.system_prompt
        user_prompt = self._ocr_prompts.ocr_user_prompt
        return self._model_wrapper.process(image, user_prompt, system_prompt)
