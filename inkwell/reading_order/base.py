from abc import ABC, abstractmethod
from typing import List, Union

from inkwell.components import Layout


class BaseReadingOrderDetector(ABC):
    @abstractmethod
    def process(
        self, layout: Union[Layout, List[Layout]]
    ) -> Union[Layout, List[Layout]]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
