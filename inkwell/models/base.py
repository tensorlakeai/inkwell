from abc import ABC, abstractmethod

import numpy as np


class BaseVisionModelWrapper(ABC):
    @abstractmethod
    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: str,
        system_prompt: str,
    ) -> list[str]:
        pass
