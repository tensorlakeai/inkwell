from pydantic import BaseModel, ConfigDict, Field


class TableDetectorConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name_hf: str = Field(..., description="HuggingFace model name")


TABLE_TRANSFORMER_TABLE_DETECTOR_CONFIG = TableDetectorConfig(
    model_name_hf="microsoft/table-transformer-detection"
)
