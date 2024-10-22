# pylint: disable=duplicate-code

import logging

import numpy as np

from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.figure_extractor.config import _load_figure_extractor_prompt
from inkwell.figure_extractor.figure_extractor import FigureExtractorType
from inkwell.ocr.phi3_ocr import Phi3VisionOCR

_logger = logging.getLogger(__name__)


class Phi3VFigureExtractor(BaseFigureExtractor):
    def __init__(self, **kwargs):
        self._prompt = _load_figure_extractor_prompt()
        self._ocr_client = Phi3VisionOCR(**kwargs)

    @property
    def model_id(self) -> str:
        return FigureExtractorType.PHI3_VISION.value

    def process(self, image_batch: list[np.ndarray]) -> list[dict]:
        _logger.info("Running Phi3 Vision Figure Extractor")
        result = self._ocr_client.process(
            image_batch=image_batch,
            user_prompt=self._prompt.user_prompt,
            system_prompt=self._prompt.system_prompt,
        )

        return result
