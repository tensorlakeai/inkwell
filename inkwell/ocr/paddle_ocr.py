# pylint: disable=duplicate-code

from typing import List, Optional, Union

import numpy as np

from inkwell.ocr.base import BaseOCR
from inkwell.ocr.ocr import OCRType
from inkwell.utils.env_utils import is_paddleocr_available

if is_paddleocr_available():
    from paddleocr import PPStructure
else:
    raise ImportError("paddleocr is not available. Please install it first.")


class PaddleOCR(BaseOCR):
    def __init__(self, **kwargs):
        self._lang = kwargs.get("lang", "en")
        self._load_engine()

    @property
    def model_id(self) -> str:
        return OCRType.PADDLE.value

    def _load_engine(self):
        self._engine = PPStructure(
            layout=True,
            table=False,
            ocr=True,
            show_log=False,
            return_ocr_result_in_table=True,
        )

    def _detect(self, image: np.ndarray) -> str:
        text_results = []
        results = self._engine(image)
        for result in results:
            for text_box in result["res"]:
                text = text_box["text"]
                text_results.append(text)

        text_str = "\n".join(text_results)
        return text_str

    def process(
        self,
        image_batch: list[np.ndarray],
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """
        Processes the image(s) and returns the text(s) detected.

        Args:
            image (np.ndarray or list[np.ndarray]): The image(s) to process.

        Returns:
            str or list[str]: The text(s) detected.
        """
        _, _ = user_prompt, system_prompt
        return [self._detect(img) for img in image_batch]
