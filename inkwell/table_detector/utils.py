# pylint: disable=duplicate-code

import pathlib

import yaml

CFG = "config.yml"


def _load_table_detector_config() -> dict:
    """
    Load the table detector configuration from the YAML file.
    """

    current_dir = pathlib.Path(__file__).parent.resolve()
    cfg_path = current_dir / CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    return model_cfg
