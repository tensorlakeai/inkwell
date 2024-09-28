import pathlib

import yaml

CFG = "models_config.yml"


def _cfg_dir():
    return pathlib.Path(__file__).parent.resolve() / CFG


def _load_models_config() -> dict:
    """
    Load the OCR configuration from the YAML file.
    """

    with open(_cfg_dir(), "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    return model_cfg
