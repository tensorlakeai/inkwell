from abc import ABC, abstractmethod

import numpy as np

from inkwell.components import Layout


class BaseTableDetector(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Layout:
        pass
