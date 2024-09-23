from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from inkwell.components import Layout


class BaseTableDetector(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Layout:
        pass


class BaseTableExtractor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Dict:
        pass
