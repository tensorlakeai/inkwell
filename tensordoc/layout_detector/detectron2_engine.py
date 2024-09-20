# pylint: disable=used-before-assignment

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from PIL import Image

from tensordoc.components import Layout, LayoutBlock, Rectangle
from tensordoc.layout_detector.base import BaseLayoutDetector, BaseLayoutEngine
from tensordoc.utils.download import download_file, get_cache_directory
from tensordoc.utils.env_utils import (
    is_detectron2_available,
    is_torch_cuda_available,
)

if is_detectron2_available():
    import detectron2.config  # pylint: disable=import-outside-toplevel
    import detectron2.engine  # pylint: disable=import-outside-toplevel


class Detectron2LayoutEngine(BaseLayoutEngine):
    """Create a Detectron2-based Layout Detection Model

    Code adapted from: https://github.com/Layout-Parser/layout-parser

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
        detection_threshold (:obj:`float`, optional):
            The threshold for the detection confidence score.
            If the score is less than the threshold,
            the detection will be ignored.
            Defaults to `0.5`.
        device(:obj:`str`, optional):
            Whether to use cuda or cpu devices. If not set, it will
            automatically determine the device to initialize the models on.

    """

    DEPENDENCIES = ["detectron2"]
    DETECTOR_NAME = "detectron2"

    def __init__(
        self,
        model_path: str,
        config_path: str,
        label_map: dict = None,
        device: Union[str, None] = None,
        add_architecture_config_method: Optional[Callable] = None,
        **kwargs,  # pylint: disable=too-many-arguments
    ):

        self._config_path = config_path
        self._model_path = model_path
        self._label_map = label_map
        self._add_architecture_config_method = add_architecture_config_method

        if device is None:
            device = "cuda" if is_torch_cuda_available() else "cpu"

        self._device = device

        self._create_cfg(**kwargs)
        self._create_model()

    def _create_cfg(self, **kwargs):
        cfg = detectron2.config.get_cfg()
        if self._add_architecture_config_method:
            self._add_architecture_config_method(cfg)

        cfg.merge_from_file(self._config_path)

        cfg.MODEL.WEIGHTS = self._model_path
        cfg.MODEL.DEVICE = self._device

        if "detection_threshold" in kwargs:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = kwargs[
                "detection_threshold"
            ]

        self._cfg = cfg

    def _create_model(self):
        self._model = detectron2.engine.DefaultPredictor(self._cfg)

    def _gather_output(self, outputs: dict) -> Layout:
        instance_pred = outputs["instances"].to("cpu")
        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            x_1, y_1, x_2, y_2 = box
            label = self._label_map.get(label, label)
            cur_block = LayoutBlock(
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


class Detectron2LayoutDetector(BaseLayoutDetector):
    """
    Detectron2-based layout detector.
    """

    def __init__(self):
        super().__init__()
        self._config = {}
        self._model = None
        self._model_name = ""

    def _load_model(self, **kwargs):

        default_model_path = (
            get_cache_directory()
            / self._model_name
            / self._config["WEIGHTS_FILE"]
        )

        model_path = kwargs.get("model_path", default_model_path)

        if not model_path.exists():

            model_path.parent.mkdir(parents=True, exist_ok=True)
            file_name = os.path.join(
                self._model_name, self._config["WEIGHTS_FILE"]
            )
            download_file(self._config["WEIGHTS_URL"], file_name)

        config_path = (
            Path(self._config["cfg_dir"]) / self._config["CONFIG_FILE"]
        )

        self._model = Detectron2LayoutEngine(
            model_path=str(model_path),
            config_path=config_path,
            label_map=self._config["LABEL_MAP"],
            **kwargs,
        )

    def process(self, image: np.ndarray) -> List[Layout]:
        return self._model.detect(image)
