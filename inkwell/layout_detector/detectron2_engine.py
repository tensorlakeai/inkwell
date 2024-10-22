# pylint: disable=used-before-assignment

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image

from inkwell.components import Layout, LayoutBlock, Rectangle
from inkwell.layout_detector.base import BaseLayoutDetector, BaseLayoutEngine
from inkwell.utils.download import download_file, get_cache_directory
from inkwell.utils.env_utils import (
    is_detectron2_available,
    is_torch_cuda_available,
)

if is_detectron2_available():
    import detectron2.config  # pylint: disable=import-outside-toplevel
    import detectron2.engine  # pylint: disable=import-outside-toplevel

_logger = logging.getLogger(__name__)


class BatchPredictor(detectron2.engine.DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images: list[np.ndarray]) -> list[dict]:
        """Run d2 on an image or a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        with torch.no_grad():
            transformed_images = []
            for image in images:
                if self.input_format == "RGB":
                    image = image[:, :, ::-1]
                height, width = image.shape[:2]
                image = self.aug.get_transform(image).apply_image(image)
                image = torch.from_numpy(
                    image.astype("float32").transpose(2, 0, 1)
                )
                image.to(self.cfg.MODEL.DEVICE)
                inputs = {"image": image, "height": height, "width": width}
                transformed_images.append(inputs)

        preds = self.model(transformed_images)
        return preds


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
        self._model = BatchPredictor(self._cfg)

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

    def detect(self, image_batch: list[np.ndarray]) -> list[Layout]:
        """Detect the layout of a given image.

        Args:
            image_batch (:obj:`np.ndarray` or `PIL.Image`):
            The input image to detect.

        Returns:
            :obj:`Layout`: The detected layout of the input image
        """

        preprocessed_images = [self.image_loader(img) for img in image_batch]
        outputs = self._model(preprocessed_images)
        return [self._gather_output(output) for output in outputs]

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

        if "model_path" in kwargs:
            model_path = (
                Path(kwargs.pop("model_path")) / self._config["WEIGHTS_FILE"]
            )
        else:
            model_path = Path(default_model_path)

        if not model_path.exists():

            model_path.parent.mkdir(parents=True, exist_ok=True)
            file_name = os.path.join(
                self._model_name, self._config["WEIGHTS_FILE"]
            )
            download_file(self._config["WEIGHTS_URL"], file_name)

        config_path = (
            Path(self._config["cfg_dir"]) / self._config["CONFIG_FILE"]
        )

        _logger.info("Loading Layout Detector model from %s", model_path)

        self._model = Detectron2LayoutEngine(
            model_path=str(model_path),
            config_path=config_path,
            label_map=self._config["LABEL_MAP"],
            **kwargs,
        )

    def process(self, image_batch: list[np.ndarray]) -> list[Layout]:
        """
        Detect the layout of a given image or a batch of images.

        Args:
            image_batch (:obj:`np.ndarray`
            or `PIL.Image`
            or `list[np.ndarray]`
            or `list[PIL.Image]`):
            The input image or list of images to detect.

        Returns:
            :obj:`Layout`
            or `list[Layout]`:
            The detected layout of the input image or list of images.
        """
        return self._model.detect(image_batch)
