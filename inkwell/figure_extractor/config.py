from pydantic import BaseModel, ConfigDict, Field

from inkwell.figure_extractor.prompts import (
    FIGURE_EXTRACTOR_SYSTEM_PROMPT,
    FIGURE_EXTRACTOR_USER_PROMPT,
)


class FigureExtractorPrompt(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt for the figure extractor"
    )
    user_prompt: str = Field(
        ..., description="User prompt for the figure extractor"
    )


class FigureExtractorConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name_hf: str = Field(..., description="HuggingFace model name")


def _load_figure_extractor_prompt() -> FigureExtractorPrompt:
    """
    Load default prompts for figure extraction tasks.
    """

    return FigureExtractorPrompt(
        system_prompt=FIGURE_EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=FIGURE_EXTRACTOR_USER_PROMPT,
    )
