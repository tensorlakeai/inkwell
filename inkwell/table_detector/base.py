from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from inkwell.components import Layout


class BaseTableDetector(ABC):
    @abstractmethod
    def process(
        self, image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[Layout, List[Layout]]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
