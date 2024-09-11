from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseOCR(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> List:
        pass
