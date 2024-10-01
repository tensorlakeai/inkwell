from abc import ABC, abstractmethod

import numpy as np


class BaseOCR(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def process(self, image: np.ndarray) -> str:
        pass
