# pylint: disable=duplicate-code

import logging
from typing import Optional

import numpy as np

from inkwell.figure_extractor.base import BaseFigureExtractor
from inkwell.figure_extractor.config import _load_figure_extractor_prompt
from inkwell.figure_extractor.figure_extractor import FigureExtractorType
from inkwell.ocr.minicpm_ocr import MiniCPMOCR
from inkwell.utils.error_handling import convert_markdown_to_json

_logger = logging.getLogger(__name__)


class MiniCPMFigureExtractor(MiniCPMOCR, BaseFigureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Call the constructor of MiniCPMOCR
        self._prompt = _load_figure_extractor_prompt()

    @property
    def model_id(self) -> str:
        return FigureExtractorType.MINI_CPM.value

    @convert_markdown_to_json
    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        _logger.info("Running MiniCPM Figure Extractor")
        if user_prompt is None:
            user_prompt = self._prompt.user_prompt
        if system_prompt is None:
            system_prompt = self._prompt.system_prompt
        return super().process(image_batch, user_prompt, system_prompt)
