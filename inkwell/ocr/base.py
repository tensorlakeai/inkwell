from abc import ABC, abstractmethod

import numpy as np


class BaseOCR(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> str:
        pass
