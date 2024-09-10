# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=unnecessary-pass

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from PIL import Image

from tensordoc.components import Layout


class BaseLayoutEngine(ABC):
    """
    Abstract class for layout detection.
    """

    @property
    @abstractmethod
    def DEPENDENCIES(self):  # pylint: disable=invalid-name
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    @property
    @abstractmethod
    def DETECTOR_NAME(self):  # pylint: disable=invalid-name
        pass

    @abstractmethod
    def detect(self, image: Union[np.ndarray, Image.Image]) -> List[Layout]:
        pass

    @abstractmethod
    def image_loader(
        self, image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        It will process the input images appropriately to the target format.
        """
        pass


class BaseLayoutDetector(ABC):
    """
    Abstract class for layout detection.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> Layout: ...
