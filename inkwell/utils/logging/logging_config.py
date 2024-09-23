import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": logging.DEBUG,
        }
    },
    "loggers": {
        "inkwell": {
            "handlers": ["console"],
            "level": logging.DEBUG,
            "propagate": False,
        }
    },
}


def setup_logging(default_level=logging.INFO):

    log_level = os.environ.get("MIN_LOG_LEVEL", default_level)
    LOGGING_CONFIG["handlers"]["console"]["level"] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)
