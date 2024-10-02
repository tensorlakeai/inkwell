import json
import logging
from functools import wraps

_logger = logging.getLogger(__name__)


def convert_markdown_to_json(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            cleaned_result = result.replace("json", "").replace("```", "")
            return json.loads(cleaned_result)
        except json.JSONDecodeError:
            _logger.warning(
                "Error decoding JSON, returning result as str: %s", result
            )
            return result

    return wrapper
