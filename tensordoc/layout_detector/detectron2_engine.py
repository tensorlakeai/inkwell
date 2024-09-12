# pylint: disable=used-before-assignment

from pathlib import Path
from typing import List, Union

import numpy as np
from detectron2.config import CfgNode as CN
from PIL import Image

from tensordoc.components import Layout, Rectangle, TextBlock
from tensordoc.layout_detector.base import BaseLayoutDetector, BaseLayoutEngine
from tensordoc.utils.env_utils import (
    is_detectron2_available,
    is_torch_cuda_available,
)

if is_detectron2_available():
    import detectron2.config  # pylint: disable=import-outside-toplevel
    import detectron2.engine  # pylint: disable=import-outside-toplevel


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    _cfg = cfg

    _cfg.MODEL.VIT = CN()

    # CoaT model name.
    _cfg.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _cfg.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _cfg.MODEL.VIT.IMG_SIZE = [224, 224]

    _cfg.MODEL.VIT.POS_TYPE = "shared_rel"

    _cfg.MODEL.VIT.DROP_PATH = 0.0

    _cfg.MODEL.VIT.MODEL_KWARGS = "{}"

    _cfg.SOLVER.OPTIMIZER = "ADAMW"

    _cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _cfg.AUG = CN()

    _cfg.AUG.DETR = False


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
        config_path: str,
        model_path: str = None,
        label_map: dict = None,
        device: Union[str, None] = None,
        **kwargs,
    ):
        self._config_path = config_path
        self._model_path = model_path
        self._label_map = label_map

        if device is None:
            device = "cuda" if is_torch_cuda_available() else "cpu"

        self._device = device

        self._create_cfg(**kwargs)
        self._create_model()

    def _create_cfg(self, **kwargs):
        cfg = detectron2.config.get_cfg()
        if kwargs.get("add_vit_config", False):
            add_vit_config(cfg)
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


class Detectron2LayoutDetector(BaseLayoutDetector):
    """
    Detectron2-based layout detector.
    """

    def __init__(self):
        super().__init__()
        self._config = {}
        self._model = None

    def _load_model(self, **kwargs):
        model_path = kwargs.get("model_path", self._config["WEIGHTS"])

        config_path = (
            Path(self._config["cfg_dir"]) / self._config["CONFIG_FILE"]
        )

        self._model = Detectron2LayoutEngine(
            model_path=model_path,
            config_path=config_path,
            label_map=self._config["LABEL_MAP"],
            **kwargs,
        )

    def process(self, image: np.ndarray) -> List[Layout]:
        return self._model.detect(image)
