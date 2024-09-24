import numpy as np

from inkwell.components import (
    Image,
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
            content=Image(
                image=np.array([[1, 2], [3, 4]]),
                text="Mock value",
            ),
        )
    ]
    return figure_fragment
