from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseOCR(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        pass
