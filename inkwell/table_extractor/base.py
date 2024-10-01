from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseTableExtractor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> Dict:
        pass
