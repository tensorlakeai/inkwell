from pydantic import BaseModel, ConfigDict, Field

from inkwell.table_extractor.prompts import (
    TABLE_EXTRACTOR_SYSTEM_PROMPT,
    TABLE_EXTRACTOR_USER_PROMPT,
)


class TableExtractorPrompt(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt for the table extractor"
    )
    user_prompt: str = Field(
        ..., description="User prompt for the table extractor"
    )


class TableExtractorConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name_hf: str = Field(..., description="HuggingFace model name")


TABLE_TRANSFORMER_TABLE_EXTRACTOR_CONFIG = TableExtractorConfig(
    model_name_hf="microsoft/table-transformer-structure-recognition"
)


def _load_table_extractor_prompt() -> TableExtractorPrompt:
    """
    Load default prompts for table extraction tasks.
    """

    return TableExtractorPrompt(
        system_prompt=TABLE_EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=TABLE_EXTRACTOR_USER_PROMPT,
    )
