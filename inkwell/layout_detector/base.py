# pylint: disable=unnecessary-pass


from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union

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
    def detect(self, image: Union[np.ndarray, Image.Image]) -> List[Layout]:
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

    @abstractmethod
    def process(self, image: np.ndarray) -> Layout:
        pass


class LayoutDetectorType(Enum):
    FASTER_RCNN = "faster_rcnn"
    DIT = "dit"
    LAYOUTLMV3 = "layoutlmv3"
