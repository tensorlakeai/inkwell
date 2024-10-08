# pylint: disable=duplicate-code

import logging

import numpy as np

from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import InferenceBackend, ModelType
from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.config import _load_table_extractor_prompt
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class Phi3VTableExtractor(BaseTableExtractor):
    def __init__(self, **kwargs):
        self._prompt = _load_table_extractor_prompt()
        self._inference_backend = kwargs.get(
            "inference_backend", InferenceBackend.VLLM
        )
        self._model_path = kwargs.get("model_path", None)
        self._model_name = (
            ModelType.PHI3_VISION_VLLM.value
            if self._inference_backend == InferenceBackend.VLLM
            else ModelType.PHI3_VISION.value
        )
        self._model_wrapper = ModelRegistry.get_model_wrapper(
            self._model_name, **{"model_path": self._model_path}
        )

    @property
    def model_id(self) -> str:
        return TableExtractorType.PHI3_VISION.value

    @convert_markdown_to_json
    def process(self, image: np.ndarray) -> dict:
        _logger.info("Running Phi3 Vision Table Extractor")
        system_prompt = self._prompt.system_prompt
        user_prompt = self._prompt.user_prompt
        result = self._model_wrapper.process(image, user_prompt, system_prompt)

        return result
