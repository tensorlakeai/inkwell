import pathlib
from typing import Tuple

import yaml

CFG = "ocr_config.yml"


def _cfg_dir():
    return pathlib.Path(__file__).parent.resolve() / CFG


def _load_ocr_config() -> dict:
    """
    Load the OCR configuration from the YAML file.
    """

    with open(_cfg_dir(), "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    return model_cfg


def _load_ocr_prompts() -> Tuple[str, str]:
    """
    Load the OCR prompts from the YAML file.
    """

    with open(_cfg_dir(), "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    return (
        model_cfg["ocr_prompts"]["system_prompt"],
        model_cfg["ocr_prompts"]["ocr_user_prompt"],
    )
