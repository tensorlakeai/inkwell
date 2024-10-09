from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np


class BaseFigureExtractor(ABC):
    @abstractmethod
    def process(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[Dict, List[Dict]]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
