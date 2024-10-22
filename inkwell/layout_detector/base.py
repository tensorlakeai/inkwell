# pylint: disable=unnecessary-pass


from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image

from inkwell.components import Layout


class BaseLayoutEngine(ABC):
    """
    Abstract class for layout detection.
    """

    @property
    @abstractmethod
    def DEPENDENCIES(self):  # pylint: disable=invalid-name
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    @property
    @abstractmethod
    def DETECTOR_NAME(self):  # pylint: disable=invalid-name
        pass

    @abstractmethod
    def detect(
        self, image_batch: Union[list[np.ndarray], list[Image.Image]]
    ) -> list[Layout]:
        pass

    @abstractmethod
    def image_loader(
        self, image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        It will process the input images appropriately to the target format.
        """
        pass


class BaseLayoutDetector(ABC):
    """
    Abstract class for layout detection.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def process(self, image_batch: list[np.ndarray]) -> list[Layout]:
        pass
