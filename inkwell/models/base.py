from abc import ABC, abstractmethod

import numpy as np


class BaseVisionModelWrapper(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, user_prompt: str, system_prompt: str
    ) -> str:
        pass
