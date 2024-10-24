from dataclasses import dataclass

import numpy as np

from inkwell.components.layout import Layout, LayoutBlock


class Config:
    arbitrary_types_allowed = True


@dataclass
class PageImage:
    page_image: np.ndarray
    page_number: int
    page_layout: Layout


@dataclass
class PageBlocks:
    page_image: np.ndarray
    figure_blocks: list[LayoutBlock]
    table_blocks: list[LayoutBlock]
    text_blocks: list[LayoutBlock]
    page_number: int


@dataclass
class DocumentPageBlocks:
    page_blocks: list[PageBlocks]
