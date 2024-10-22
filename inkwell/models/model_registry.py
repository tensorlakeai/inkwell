import logging
from typing import Callable

from inkwell.models.base import BaseVisionModelWrapper

_logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance = None
    _model_wrappers: dict[str, BaseVisionModelWrapper] = {}
    _wrapper_loaders: dict[str, Callable] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_wrapper_loader(cls, model_name: str, loader: Callable):
        _logger.info("Registering loader for model: %s", model_name)
        cls._wrapper_loaders[model_name] = loader

    @classmethod
    def get_model_wrapper(cls, model_name: str, **kwargs):
        if model_name not in cls._model_wrappers:
            if model_name not in cls._wrapper_loaders:
                raise ValueError(
                    f"No loader registered for model: {model_name}"
                )
            cls._model_wrappers[model_name] = cls._wrapper_loaders[model_name](
                **kwargs
            )
        return cls._model_wrappers[model_name]
