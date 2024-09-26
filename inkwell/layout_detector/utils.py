import pathlib

import yaml

from inkwell.layout_detector.layout_detector import LayoutDetectorType

CFG_DIR = "config"
LAYOUT_DETECTOR_CONFIG_FILE = "layout_detector_configurations.yml"


def load_layout_detector_config(
    layout_detector_type: LayoutDetectorType,
) -> dict:
    """
    Load the layout detector configuration from the YAML file.
    """

    current_dir = pathlib.Path(__file__).parent.resolve()
    cfg_path = current_dir / CFG_DIR / LAYOUT_DETECTOR_CONFIG_FILE
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg = model_cfg[layout_detector_type.value]

    model_cfg["cfg_dir"] = str(current_dir / CFG_DIR)
    return model_cfg
