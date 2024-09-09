import yaml

LAYOUT_DETECTOR_CONFIG_PATH = "tensordoc/config/layout_detector.yml"


def load_layout_detector_config() -> dict:
    with open(LAYOUT_DETECTOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
