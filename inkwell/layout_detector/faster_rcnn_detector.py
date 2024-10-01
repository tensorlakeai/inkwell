from inkwell.layout_detector.detectron2_engine import Detectron2LayoutDetector
from inkwell.layout_detector.layout_detector import LayoutDetectorType
from inkwell.layout_detector.utils import load_layout_detector_config


class FasterRCNNLayoutDetector(Detectron2LayoutDetector):
    """
    Faster RCNN based layout detector using Detectron2.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._model_name = "faster_rcnn"
        model_cfg = load_layout_detector_config(LayoutDetectorType.FASTER_RCNN)
        self._config = model_cfg
        self._load_model(**kwargs)

    @property
    def model_id(self) -> str:
        return LayoutDetectorType.FASTER_RCNN.value
