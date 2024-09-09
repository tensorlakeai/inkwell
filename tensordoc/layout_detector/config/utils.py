import pathlib
from typing import Tuple

import yaml


def load_layout_detector_config() -> Tuple[dict, str]:
    with open(
        pathlib.Path(__file__).parent.resolve() / "layout_detector.yml",
        "r",
        encoding="utf-8",
    ) as f:
        model_cfg = yaml.safe_load(f)

    detectron_cfg_path = (
        pathlib.Path(__file__).parent.resolve()
        / "faster_rcnn_detectron_cfg.yml"
    )

    return model_cfg, str(detectron_cfg_path)
