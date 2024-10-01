import numpy as np
from PIL import Image as PILImage

from inkwell.components import (
    Figure,
    PageFragment,
    PageFragmentType,
    Table,
    TableEncoding,
    TextBox,
)


def get_mock_text_fragment():
    text_fragment = [
        PageFragment(
            fragment_type=PageFragmentType.TEXT,
            content=TextBox(
                text="Mock value",
            ),
        )
    ]
    return text_fragment


def get_mock_table_fragment():
    table_fragment = [
        PageFragment(
            fragment_type=PageFragmentType.TABLE,
            content=Table(
                data={"Mock value": ["Mock value"]},
                encoding=TableEncoding.DICT,
            ),
        )
    ]
    return table_fragment


def get_mock_figure_fragment():
    figure_fragment = [
        PageFragment(
            fragment_type=PageFragmentType.FIGURE,
            content=Figure(
                image=PILImage.fromarray(np.zeros((5, 5))),
                text="Mock value",
            ),
        )
    ]
    return figure_fragment
