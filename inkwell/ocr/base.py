from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseOCR(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def process(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        pass
