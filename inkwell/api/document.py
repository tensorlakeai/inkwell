from typing import Any, Dict, List

from pydantic import BaseModel

from inkwell.api.page import Page


class Document(BaseModel):
    """
    Document in a document.
    """

    pages: List[Page]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": [page.to_dict() for page in self.pages],
        }
