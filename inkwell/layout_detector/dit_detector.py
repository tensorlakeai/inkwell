from detectron2.config import CfgNode as CN

from inkwell.layout_detector.base import LayoutDetectorType
from inkwell.layout_detector.detectron2_engine import Detectron2LayoutDetector
from inkwell.layout_detector.dit.backbone import (  # noqa pylint: disable=unused-import
    build_vit_fpn_backbone,
)
from inkwell.layout_detector.utils import load_layout_detector_config


class DitLayoutDetector(Detectron2LayoutDetector):
    """
    DiT based layout detector using Detectron2.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._model_name = "dit"

        model_cfg = load_layout_detector_config(LayoutDetectorType.DIT)
        self._config = model_cfg

        kwargs["add_architecture_config_method"] = self._add_vit_config
        self._load_model(**kwargs)

    # pylint: disable=duplicate-code
    def _add_vit_config(self, cfg):
        """
        Add config for VIT specific to DiT.
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
