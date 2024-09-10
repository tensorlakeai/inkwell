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

# pylint: disable=used-before-assignment

from typing import Union

import numpy as np
from PIL import Image

from tensordoc.components import Layout, Rectangle, TextBlock
from tensordoc.layout_detector.base_layout_engine import BaseLayoutEngine
from tensordoc.utils.env_utils import (
    is_detectron2_available,
    is_torch_cuda_available,
)

if is_detectron2_available():
    import detectron2.config  # pylint: disable=import-outside-toplevel
    import detectron2.engine  # pylint: disable=import-outside-toplevel


class Detectron2LayoutEngine(BaseLayoutEngine):
    """Create a Detectron2-based Layout Detection Model

    Args:
        config_path (:obj:`str`):
            The path to the configuration file.
        model_path (:obj:`str`, None):
            The path to the saved weights of the model.
            If set, overwrite the weights in the configuration file.
            Defaults to `None`.
        label_map (:obj:`dict`, optional):
            The map from the model prediction (ids) to real
            word labels (strings). If the config is from one of the
            supported datasets, Layout Parser will
            automatically initialize the label_map.
            Defaults to `None`.
        device(:obj:`str`, optional):
            Whether to use cuda or cpu devices. If not set, it will
            automatically determine the device to initialize the models on.
    """

    DEPENDENCIES = ["detectron2"]
    DETECTOR_NAME = "detectron2"

    def __init__(
        self,
        config_path: str,
        model_path: str = None,
        label_map: dict = None,
        device: Union[str, None] = None,
    ):

        cfg = detectron2.config.get_cfg()
        cfg.merge_from_file(config_path)

        cfg.MODEL.WEIGHTS = model_path

        if device is None:
            if is_torch_cuda_available():
                device = "cuda"
            else:
                device = "cpu"

        cfg.MODEL.DEVICE = device

        self.cfg = cfg
        self.label_map = label_map
        self._create_model()

    def _create_model(self):
        self._model = detectron2.engine.DefaultPredictor(self.cfg)

    def _gather_output(self, outputs: dict) -> Layout:

        instance_pred = outputs["instances"].to("cpu")
        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            x_1, y_1, x_2, y_2 = box
            label = self.label_map.get(label, label)
            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray` or `PIL.Image`):
            The input image to detect.

        Returns:
            :obj:`Layout`: The detected layout of the input image
        """

        image = self.image_loader(image)
        outputs = self._model(image)
        layout = self._gather_output(outputs)
        return layout

    def image_loader(
        self, image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        return image

    def __new__(cls, *args, **kwargs) -> "Detectron2LayoutEngine":
        if not is_detectron2_available():
            raise ImportError(
                "Detectron2 is not installed. Please install it first."
            )
        return super().__new__(cls)
