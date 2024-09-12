from tensordoc.layout_detector.base import LayoutDetectorType
from tensordoc.layout_detector.detectron2_engine import (
    Detectron2LayoutDetector,
)
from tensordoc.layout_detector.dit.backbone import (  # noqa pylint: disable=unused-import
    build_vit_fpn_backbone,
)
from tensordoc.layout_detector.utils import load_layout_detector_config


class DitLayoutDetector(Detectron2LayoutDetector):
    """
    DiT based layout detector using Detectron2.
    """

    def __init__(self, **kwargs):
        super().__init__()
        kwargs["add_vit_config"] = True

        model_cfg = load_layout_detector_config(LayoutDetectorType.DIT)
        self._config = model_cfg
        self._load_model(**kwargs)
