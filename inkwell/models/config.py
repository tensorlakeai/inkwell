from pydantic import BaseModel, ConfigDict, Field


class GenerationArgs(BaseModel):
    max_new_tokens: int = Field(
        2048, description="Maximum number of new tokens to generate"
    )
    temperature: float = Field(
        None, description="Temperature for text generation"
    )
    top_p: float = Field(None, description="Top-p value for text generation")
    top_k: int = Field(None, description="Top-k value for text generation")


class ModelGenerationConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name_hf: str = Field(..., description="HuggingFace model name")
    generation_args: GenerationArgs = Field(
        ..., description="Generation arguments"
    )


PHI3_VISION_MODEL_CONFIG = ModelGenerationConfig(
    model_name_hf="microsoft/Phi-3.5-vision-instruct",
    generation_args=GenerationArgs(
        max_new_tokens=2048, temperature=0.2, top_p=0.95
    ),
)

QWEN2_VISION_MODEL_CONFIG = ModelGenerationConfig(
    model_name_hf="Qwen/Qwen2-VL-2B-Instruct",
    generation_args=GenerationArgs(
        max_new_tokens=2048, temperature=0.2, top_p=0.95, top_k=-1
    ),
)

MINI_CPM_MODEL_CONFIG = ModelGenerationConfig(
    model_name_hf="openbmb/MiniCPM-V-2_6",
    generation_args=GenerationArgs(
        max_new_tokens=2048, temperature=0.2, top_p=0.95, top_k=-1
    ),
)
