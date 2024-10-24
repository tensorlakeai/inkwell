# flake8: noqa: E501

from inkwell.figure_extractor.figure_extractor import FigureExtractorType
from inkwell.figure_extractor.minicpm_figure_extractor import (
    MiniCPMFigureExtractor,
)
from inkwell.figure_extractor.openai_4o_mini_figure_extractor import (
    OpenAI4OMiniFigureExtractor,
)
from inkwell.figure_extractor.phi3v_figure_extractor import (
    Phi3VFigureExtractor,
)
from inkwell.utils.env_utils import is_vllm_available


class FigureExtractorFactory:
    @staticmethod
    def get_figure_extractor(
        figure_extractor_type: FigureExtractorType, **kwargs
    ):
        if figure_extractor_type == FigureExtractorType.OPENAI_GPT4O_MINI:
            return OpenAI4OMiniFigureExtractor()
        if figure_extractor_type == FigureExtractorType.PHI3_VISION:
            if is_vllm_available():
                return Phi3VFigureExtractor(**kwargs)
            raise ValueError(
                "Please install vllm to use Phi3 Vision Figure Extractor"
            )
        if figure_extractor_type == FigureExtractorType.MINI_CPM:
            if is_vllm_available():
                return MiniCPMFigureExtractor(**kwargs)
            raise ValueError(
                "Please install vllm to use MiniCPM Figure Extractor"
            )
        raise ValueError(
            f"Invalid figure extractor type: {figure_extractor_type}"
        )
