from abc import ABC, abstractmethod

from inkwell.components import Layout


class BaseReadingOrderDetector(ABC):
    @abstractmethod
    def process(self, layout: list[Layout]) -> list[Layout]:
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass
