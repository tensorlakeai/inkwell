# pylint: disable=duplicate-code

import html_to_json
import numpy as np

from inkwell.table_extractor.base import BaseTableExtractor
from inkwell.table_extractor.table_extractor import TableExtractorType
from inkwell.utils.env_utils import is_paddleocr_available

if is_paddleocr_available():
    from paddleocr import PPStructure
else:
    raise ImportError("paddleocr is not available. Please install it first.")


class PaddleTableExtractor(BaseTableExtractor):
    def __init__(self):
        self._load_engine()

    @property
    def model_id(self) -> str:
        return TableExtractorType.PADDLE.value

    def _load_engine(self):
        self._engine = PPStructure(
            layout=False,
            table=True,
            ocr=False,
            show_log=False,
            return_ocr_result_in_table=True,
            lang="en",
        )

    def _detect(self, image: np.ndarray) -> dict:
        results = self._engine(image)
        table_html = results[0]["res"]["html"]
        table_json = html_to_json.convert_tables(table_html)
        return {"data": table_json}

    def process(self, image_batch: list[np.ndarray]) -> list[dict]:
        return [self._detect(img) for img in image_batch]
