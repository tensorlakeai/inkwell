from pydantic import BaseModel, ConfigDict, Field

from inkwell.ocr.prompts import (
    OCR_SYSTEM_PROMPT_DEFAULT,
    OCR_USER_PROMPT_DEFAULT,
)


class OCRPrompts(BaseModel):
    system_prompt: str = Field(
        ..., description="Default system prompt for OCR tasks"
    )
    ocr_user_prompt: str = Field(
        ..., description="Default user prompt for OCR tasks"
    )


class OpenAIOCRModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name_openai: str = Field(..., description="OpenAI model name")


OPENAI_OCR_MODEL_CONFIG = OpenAIOCRModelConfig(model_name_openai="gpt-4o-mini")


def _load_ocr_prompts() -> OCRPrompts:
    """
    Load default prompts for OCR tasks.
    """

    return OCRPrompts(
        system_prompt=OCR_SYSTEM_PROMPT_DEFAULT,
        ocr_user_prompt=OCR_USER_PROMPT_DEFAULT,
    )
