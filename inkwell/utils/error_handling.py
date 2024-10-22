import json
import logging
from functools import wraps

_logger = logging.getLogger(__name__)


def convert_markdown_to_json(func):

    def _clean_string(example: str) -> str:
        return example.replace("json", "").replace("```", "")

    def _load_json(result: list[str]) -> list[str]:
        cleaned_results = []
        for res in result:
            try:
                cleaned_result = _clean_string(res)
                cleaned_results.append(json.loads(cleaned_result))
            except json.JSONDecodeError:
                _logger.warning(
                    "Error decoding JSON, returning result as str: %s", res
                )
                cleaned_results.append(res)
        return cleaned_results

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _load_json(result)

    return wrapper
