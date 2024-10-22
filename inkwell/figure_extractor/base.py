from abc import ABC, abstractmethod

import numpy as np


class BaseFigureExtractor(ABC):
    @abstractmethod
    def process(self, image_batch: list[np.ndarray]) -> list[dict]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
