import pathlib

import yaml

CFG = "config.yml"

TABLE_EXTRACTOR_PROMPT = """Extract information from the table image
into the following json format:

{
    "header": List[str] # list of header names
    "data": List[List[str]] # list of rows, each row is a list of strings
}

Strictly return the json output only, and nothing else."""


def _load_table_extractor_config() -> dict:
    """
    Load the table detector configuration from the YAML file.
    """

    current_dir = pathlib.Path(__file__).parent.resolve()
    cfg_path = current_dir / CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    return model_cfg
