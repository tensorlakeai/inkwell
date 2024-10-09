import logging

import numpy as np

from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.figure_extractor.config import _load_figure_extractor_prompt
from inkwell.figure_extractor.figure_extractor import FigureExtractorType
from inkwell.ocr.openai_4o_mini_ocr import OpenAI4OMiniOCR

_logger = logging.getLogger(__name__)


class OpenAI4OMiniFigureExtractor(BaseFigureExtractor):
    def __init__(self):
        self._load_client()
        self._prompt = _load_figure_extractor_prompt()

    @property
    def model_id(self) -> str:
        return FigureExtractorType.OPENAI_GPT4O_MINI.value

    def _load_client(self):
        self._client = OpenAI4OMiniOCR()

    def process(self, image: np.ndarray) -> str:
        if isinstance(image, list):
            results = [self._process_image(img) for img in image]
            return results
        return self._process_image(image)

    def _process_image(self, image: np.ndarray) -> dict:
        _logger.info("Running OpenAI GPT-4o Mini Figure Extractor")
        result = self._client.process(
            image,
            user_prompt=self._prompt.user_prompt,
            system_prompt=self._prompt.system_prompt,
        )

        return result
