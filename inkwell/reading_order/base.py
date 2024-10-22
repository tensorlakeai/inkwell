from abc import ABC, abstractmethod

import numpy as np

from inkwell.components import Layout


class BaseReadingOrderDetector(ABC):
    @abstractmethod
    def process(
        self, image_batch: list[np.ndarray], layout_batch: list[Layout]
    ) -> list[Layout]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
