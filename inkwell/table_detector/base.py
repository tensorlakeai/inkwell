from abc import ABC, abstractmethod

import numpy as np

from inkwell.components import Layout


class BaseTableDetector(ABC):
    @abstractmethod
    def process(self, image: list[np.ndarray]) -> list[Layout]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
