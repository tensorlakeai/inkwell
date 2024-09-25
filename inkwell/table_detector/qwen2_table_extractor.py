from PIL import Image

try:
    from inkwell.ocr.qwen2_ocr import Qwen2VisionOCR
except ImportError:
    print("Please install the latest transformers from source to use Qwen2 Vision OCR")

from inkwell.table_detector.base import BaseTableExtractor
from inkwell.table_detector.utils import TABLE_EXTRACTOR_PROMPT


class Qwen2TableExtractor(Qwen2VisionOCR, BaseTableExtractor):
    def __init__(self):
        super().__init__(user_prompt=TABLE_EXTRACTOR_PROMPT)

    def process(self, image: Image.Image) -> str:
        extracted_text = super().process(image)
        return extracted_text
