import json
import logging
from functools import wraps

_logger = logging.getLogger(__name__)


def convert_markdown_to_json(func):

    def _load_json(result: str) -> str:
        try:
            cleaned_result = result.replace("json", "").replace("```", "")
            return json.loads(cleaned_result)
        except json.JSONDecodeError:
            _logger.warning(
                "Error decoding JSON, returning result as str: %s", result
            )
            return result

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, list):
            return [_load_json(item) for item in result]
        return _load_json(result)

    return wrapper
