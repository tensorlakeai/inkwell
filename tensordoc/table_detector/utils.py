import pathlib

import yaml

from tensordoc.table_detector.table_detector import TableDetectorType

CFG = "config.yml"


def load_table_detector_config(
    table_detector_type: TableDetectorType,
) -> dict:
    """
    Load the table detector configuration from the YAML file.
    """

    current_dir = pathlib.Path(__file__).parent.resolve()
    cfg_path = current_dir / CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg = model_cfg[table_detector_type.value]

    return model_cfg
