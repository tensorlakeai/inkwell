from typing import Union

import numpy as np
from IPython.display import display
from PIL import Image


def visualize_image(image: Union[np.ndarray, Image.Image]):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, "RGB")

    display(image)
