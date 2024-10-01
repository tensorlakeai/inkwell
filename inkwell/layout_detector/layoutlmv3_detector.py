from detectron2.config import CfgNode as CN

from inkwell.layout_detector.detectron2_engine import Detectron2LayoutDetector
from inkwell.layout_detector.layout_detector import LayoutDetectorType
from inkwell.layout_detector.layoutlmv3.backbone import (  # noqa pylint: disable=unused-import
    build_vit_fpn_backbone_layoutlmv3,
)
from inkwell.layout_detector.layoutlmv3.rcnn_vl import (  # noqa pylint: disable=unused-import
    VLGeneralizedRCNN,
)
from inkwell.layout_detector.utils import load_layout_detector_config


class LayoutLMv3Detector(Detectron2LayoutDetector):
    """
    LayoutLMv3 based layout detector using Detectron2.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._model_name = "layoutlmv3"

        model_cfg = load_layout_detector_config(LayoutDetectorType.LAYOUTLMV3)
        self._config = model_cfg

        kwargs["add_architecture_config_method"] = self._add_vit_config
        self._load_model(**kwargs)

    def _add_vit_config(self, cfg):
        """
        Add config for VIT specific to LayoutLMv3.
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

        _cfg.MODEL.IMAGE_ONLY = True
        _cfg.PUBLAYNET_DATA_DIR_TRAIN = ""
        _cfg.PUBLAYNET_DATA_DIR_TEST = ""
        _cfg.FOOTNOTE_DATA_DIR_TRAIN = ""
        _cfg.FOOTNOTE_DATA_DIR_VAL = ""
        _cfg.SCIHUB_DATA_DIR_TRAIN = ""
        _cfg.SCIHUB_DATA_DIR_TEST = ""
        _cfg.JIAOCAI_DATA_DIR_TRAIN = ""
        _cfg.JIAOCAI_DATA_DIR_TEST = ""
        _cfg.ICDAR_DATA_DIR_TRAIN = ""
        _cfg.ICDAR_DATA_DIR_TEST = ""
        _cfg.M6DOC_DATA_DIR_TEST = ""
        _cfg.DOCSTRUCTBENCH_DATA_DIR_TEST = ""
        _cfg.DOCSTRUCTBENCHv2_DATA_DIR_TEST = ""
        _cfg.CACHE_DIR = ""
        _cfg.MODEL.CONFIG_PATH = ""

        # effective update steps would be MAX_ITER/GRADIENT_ACCUMULATION_STEPS
        # maybe need to set MAX_ITER *= GRADIENT_ACCUMULATION_STEPS
        _cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    @property
    def model_id(self) -> str:
        return LayoutDetectorType.LAYOUTLMV3.value
