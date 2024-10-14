# flake8: noqa: E501

from inkwell.ocr.ocr import OCRType
from inkwell.ocr.openai_4o_mini_ocr import OpenAI4OMiniOCR
from inkwell.ocr.tesseract_ocr import TesseractOCR
from inkwell.utils.env_utils import (
    is_paddleocr_available,
    is_qwen2_available,
    is_vllm_available,
)


class OCRFactory:
    @staticmethod
    def get_ocr(ocr_type: OCRType, **kwargs):
        if ocr_type == OCRType.TESSERACT:
            return TesseractOCR(**kwargs)

        if ocr_type == OCRType.PHI3_VISION:
            if is_vllm_available():
                from inkwell.ocr.phi3_ocr import (  # pylint: disable=import-outside-toplevel
                    Phi3VisionOCR,
                )

                return Phi3VisionOCR(**kwargs)

        if ocr_type == OCRType.MINI_CPM:
            if is_vllm_available():
                from inkwell.ocr.minicpm_ocr import (  # pylint: disable=import-outside-toplevel
                    MiniCPMOCR,
                )

                return MiniCPMOCR(**kwargs)

        if ocr_type == OCRType.OPENAI_GPT4O_MINI:
            return OpenAI4OMiniOCR()
        if ocr_type == OCRType.QWEN2_2B_VISION:
            if is_qwen2_available() and is_vllm_available():
                from inkwell.ocr.qwen2_ocr import (  # pylint: disable=import-outside-toplevel
                    Qwen2OCR,
                )

                return Qwen2OCR(**kwargs)
            raise ValueError(
                "Please install the latest transformers from \
                    source to use Qwen2 Vision OCR"
            )
        if ocr_type == OCRType.PADDLE:
            if is_paddleocr_available():
                from inkwell.ocr.paddle_ocr import (  # pylint: disable=import-outside-toplevel
                    PaddleOCR,
                )

                return PaddleOCR(**kwargs)
            raise ValueError("Please install paddleocr to use PaddleOCR")
        raise ValueError(f"Invalid OCR type: {ocr_type}")
