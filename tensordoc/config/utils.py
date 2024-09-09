import pathlib

import yaml


def load_layout_detector_config() -> dict:
    with open(
        pathlib.Path(__file__).parent.resolve() / "layout_detector.yml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)
