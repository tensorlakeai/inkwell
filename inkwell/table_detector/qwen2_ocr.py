from PIL import Image

from inkwell.ocr.qwen2_ocr import Qwen2VisionOCR
from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.utils import TABLE_EXTRACTOR_PROMPT


class Qwen2TableExtractor(Qwen2VisionOCR, BaseTableExtractor):
    def __init__(self):
        super().__init__(user_prompt=TABLE_EXTRACTOR_PROMPT)

    def process(self, image: Image.Image) -> str:
        extracted_text = super().process(image)
        return extracted_text
