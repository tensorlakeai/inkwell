from inkwell.ocr.ocr import OCRType
from inkwell.ocr.phi3_ocr import Phi3VisionOCR
from inkwell.ocr.tesseract_ocr import TesseractOCR

try:
    from inkwell.ocr.qwen2_ocr import Qwen2VisionOCR
except ImportError:
    print("Please install the latest transformers from source to use Qwen2 Vision OCR")

class OCRFactory:
    @staticmethod
    def get_ocr(ocr_type: OCRType, **kwargs):
        if ocr_type == OCRType.TESSERACT:
            return TesseractOCR(**kwargs)
        if ocr_type == OCRType.PHI3_VISION:
            return Phi3VisionOCR(**kwargs)
        if ocr_type == OCRType.QWEN2_VISION:
            return Qwen2VisionOCR(**kwargs)
        raise ValueError(f"Invalid OCR type: {ocr_type}")
