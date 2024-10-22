import logging

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams

from inkwell.models.base import BaseVisionModelWrapper
from inkwell.models.config import MINI_CPM_MODEL_CONFIG
from inkwell.models.model_registry import ModelRegistry
from inkwell.models.models import ModelType
from inkwell.utils.env_utils import (
    is_flash_attention_available,
    is_torch_cuda_available,
    is_vllm_available,
)

_logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "(<image>./</image>)"
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


class MiniCPMModelWrapperVLLM(BaseVisionModelWrapper):

    def __init__(self, **kwargs):
        self._model_cfg = MINI_CPM_MODEL_CONFIG
        self._model_path = kwargs.get("model_path", None)
        self._load_model()

    def _load_model(self):
        if not (
            is_torch_cuda_available()
            and is_flash_attention_available()
            and is_vllm_available()
        ):
            raise ValueError(
                "vLLM based models work best with \
            flash-attention"
            )

        model_path = self._model_path or self._model_cfg.model_name_hf

        _logger.info("Loading model from path: %s", model_path)

        self._model = LLM(
            model=model_path, trust_remote_code=True, max_model_len=10000
        )

    def _get_messages(
        self, system_prompt: str, image_placeholder: str, user_prompt: str
    ):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_placeholder + user_prompt},
        ]

    def _preprocess_input(self, system_prompt: str, user_prompt: str) -> str:
        messages = self._get_messages(
            system_prompt, IMAGE_PLACEHOLDER, user_prompt
        )
        inputs = self._model.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return inputs

    def _load_generation_args(self):
        generation_args = dict(MINI_CPM_MODEL_CONFIG.generation_args)
        if "max_new_tokens" in generation_args:
            generation_args["max_tokens"] = generation_args.pop(
                "max_new_tokens"
            )

        generation_args = {
            k: v for k, v in generation_args.items() if v is not None
        }

        return generation_args

    def _stop_tokens(self):
        return [
            self._model.get_tokenizer().convert_tokens_to_ids(i)
            for i in STOP_TOKENS
        ]

    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: str,
        system_prompt: str,
    ) -> list[str]:

        prompts = self._preprocess_input(system_prompt, user_prompt)
        generation_args = self._load_generation_args()
        sampling_params = SamplingParams(
            **generation_args, stop_token_ids=self._stop_tokens()
        )

        data = [
            {
                "prompt": prompts,
                "multi_modal_data": {"image": Image.fromarray(img)},
            }
            for img in image_batch
        ]
        outputs = self._model.generate(data, sampling_params=sampling_params)

        return [output.outputs[0].text for output in outputs]


# Register the MiniCPM model loader
ModelRegistry.register_wrapper_loader(
    model_name=ModelType.MINI_CPM_VLLM.value,
    loader=MiniCPMModelWrapperVLLM,
)
