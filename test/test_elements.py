# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa

import unittest

import numpy as np
import pytest

from inkwell.components import (
    Interval,
    Layout,
    LayoutBlock,
    Quadrilateral,
    Rectangle,
)
from inkwell.components.elements import (
    InvalidShapeError,
    NotSupportedShapeError,
)


class TestElements(unittest.TestCase):
    def test_interval(self):

        i = Interval(1, 2, axis="y", canvas_height=30, canvas_width=400)
        i.to_rectangle()
        i.to_quadrilateral()
        assert i.shift(1) == Interval(
            2, 3, axis="y", canvas_height=30, canvas_width=400
        )
        assert i.area == 1 * 400

        i = Interval(1, 2, axis="x")
        assert i.shift([1, 2]) == Interval(2, 3, axis="x")
        assert i.scale([2, 1]) == Interval(2, 4, axis="x")
        assert i.pad(left=10, right=20) == Interval(
            0, 22, axis="x"
        )  # Test the safe_mode
        assert i.pad(left=10, right=20, safe_mode=False) == Interval(
            -9, 22, axis="x"
        )
        assert i.area == 0

        img = np.random.randint(12, 24, (40, 40))
        img[:, 10:20] = 0
        i = Interval(5, 11, axis="x")
        assert np.unique(i.crop_image(img)[:, -1]) == np.array([0])

    def test_rectangle(self):

        r = Rectangle(1, 2, 3, 4)
        r.to_interval(axis="x")
        r.to_quadrilateral()
        assert r.pad(left=1, right=5, top=2, bottom=4) == Rectangle(0, 0, 8, 8)
        assert r.shift([1, 2]) == Rectangle(2, 4, 4, 6)
        assert r.shift(1) == Rectangle(2, 3, 4, 5)
        assert r.scale([3, 2]) == Rectangle(3, 4, 9, 8)
        assert r.scale(2) == Rectangle(2, 4, 6, 8)
        assert r.area == 4

        img = np.random.randint(12, 24, (40, 40))
        assert r.crop_image(img).shape == (2, 2)

    def test_quadrilateral(self):

        points = np.array([[2, 2], [6, 2], [6, 7], [2, 6]])
        q = Quadrilateral(points)
        q.to_interval(axis="x")
        q.to_rectangle()
        assert q.shift(1) == Quadrilateral(points + 1)
        assert q.shift([1, 2]) == Quadrilateral(points + np.array([1, 2]))
        assert q.scale(2) == Quadrilateral(points * 2)
        assert q.scale([3, 2]) == Quadrilateral(points * np.array([3, 2]))
        assert q.pad(left=1, top=2, bottom=4) == Quadrilateral(
            np.array([[1, 0], [6, 0], [6, 11], [1, 10]])
        )
        assert (
            q.mapped_rectangle_points
            == np.array([[0, 0], [4, 0], [4, 5], [0, 5]])
        ).all()

        points = np.array([[2, 2], [6, 2], [6, 5], [2, 5]])
        q = Quadrilateral(points)
        img = np.random.randint(2, 24, (30, 20)).astype("uint8")
        img[2:5, 2:6] = 0
        assert np.unique(q.crop_image(img)) == np.array([0])

        q = Quadrilateral(np.array([[-2, 0], [0, 2], [2, 0], [0, -2]]))
        assert q.area == 8.0

        q = Quadrilateral([1, 2, 3, 4, 5, 6, 7, 8])
        assert (q.points == np.array([[1, 2], [3, 4], [5, 6], [7, 8]])).all()

        q = Quadrilateral([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert (q.points == np.array([[1, 2], [3, 4], [5, 6], [7, 8]])).all()

        with pytest.raises(ValueError):
            Quadrilateral([1, 2, 3, 4, 5, 6, 7])  # Incompatible list length

        with pytest.raises(ValueError):
            Quadrilateral(
                np.array([[2, 2], [6, 2], [6, 5]])
            )  # Incompatible ndarray shape

    def test_interval_relations(self):

        i = Interval(4, 5, axis="y")
        r = Rectangle(3, 3, 5, 6)
        q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))

        assert i.is_in(i)
        assert i.is_in(r)
        assert i.is_in(q)

        # convert to absolute then convert back to relative
        assert i.condition_on(i).relative_to(i) == i
        assert (
            i.condition_on(r).relative_to(r)
            == i.put_on_canvas(r).to_rectangle()
        )
        assert (
            i.condition_on(q).relative_to(q)
            == i.put_on_canvas(q).to_quadrilateral()
        )

        # convert to relative then convert back to absolute
        assert i.relative_to(i).condition_on(i) == i
        assert (
            i.relative_to(r).condition_on(r)
            == i.put_on_canvas(r).to_rectangle()
        )
        assert (
            i.relative_to(q).condition_on(q)
            == i.put_on_canvas(q).to_quadrilateral()
        )

    def test_rectangle_relations(self):

        i = Interval(4, 5, axis="y")
        q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
        r = Rectangle(3, 3, 5, 6)

        assert not r.is_in(q)
        assert r.is_in(q, soft_margin={"bottom": 1})
        assert r.is_in(q.to_rectangle())
        assert r.is_in(q.to_interval(axis="x"))

        # convert to absolute then convert back to relative
        assert r.condition_on(i).relative_to(i) == r
        assert r.condition_on(r).relative_to(r) == r
        assert r.condition_on(q).relative_to(q) == r.to_quadrilateral()

        # convert to relative then convert back to absolute
        assert r.relative_to(i).condition_on(i) == r
        assert r.relative_to(r).condition_on(r) == r
        assert r.relative_to(q).condition_on(q) == r.to_quadrilateral()

    def test_quadrilateral_relations(self):

        i = Interval(4, 5, axis="y")
        q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
        r = Rectangle(3, 3, 5, 6)

        assert not q.is_in(r)
        assert q.is_in(i, soft_margin={"top": 2, "bottom": 2})
        assert q.is_in(
            r, soft_margin={"left": 1, "top": 1, "right": 1, "bottom": 1}
        )
        assert q.is_in(q)

        # convert to absolute then convert back to relative
        assert q.condition_on(i).relative_to(i) == q
        assert q.condition_on(r).relative_to(r) == q
        assert q.condition_on(q).relative_to(q) == q

        # convert to relative then convert back to absolute
        assert q.relative_to(i).condition_on(i) == q
        assert q.relative_to(r).condition_on(r) == q
        assert q.relative_to(q).condition_on(q) == q

    def test_layoutblock(self):

        i = Interval(4, 5, axis="y")
        q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
        r = Rectangle(3, 3, 5, 6)

        t = LayoutBlock(i, id=1, type=2, text="12")
        assert (
            t.relative_to(q).condition_on(q).block
            == i.put_on_canvas(q).to_quadrilateral()
        )
        t = LayoutBlock(r, id=1, type=2, parent="a")
        assert t.relative_to(i).condition_on(i).block == r
        t = LayoutBlock(q, id=1, type=2, parent="a")
        assert t.relative_to(r).condition_on(r).block == q

        # Ensure the operations did not change the object itself
        assert t == LayoutBlock(q, id=1, type=2, parent="a")
        t1 = LayoutBlock(q, id=1, type=2, parent="a")
        t2 = LayoutBlock(i, id=1, type=2, text="12")
        t1.relative_to(t2)
        assert t2.is_in(t1)

        t = LayoutBlock(q, score=0.2)

        # Additional test for shape conversion
        assert LayoutBlock(
            i, id=1, type=2, text="12"
        ).to_interval() == LayoutBlock(i, id=1, type=2, text="12")
        assert LayoutBlock(
            i, id=1, type=2, text="12"
        ).to_rectangle() == LayoutBlock(
            i.to_rectangle(), id=1, type=2, text="12"
        )
        assert LayoutBlock(
            i, id=1, type=2, text="12"
        ).to_quadrilateral() == LayoutBlock(
            i.to_quadrilateral(), id=1, type=2, text="12"
        )

        assert LayoutBlock(r, id=1, type=2, parent="a").to_interval(
            axis="x"
        ) == LayoutBlock(r.to_interval(axis="x"), id=1, type=2, parent="a")
        assert LayoutBlock(r, id=1, type=2, parent="a").to_interval(
            axis="y"
        ) == LayoutBlock(r.to_interval(axis="y"), id=1, type=2, parent="a")
        assert LayoutBlock(
            r, id=1, type=2, parent="a"
        ).to_rectangle() == LayoutBlock(r, id=1, type=2, parent="a")
        assert LayoutBlock(
            r, id=1, type=2, parent="a"
        ).to_quadrilateral() == LayoutBlock(
            r.to_quadrilateral(), id=1, type=2, parent="a"
        )

        assert LayoutBlock(q, id=1, type=2, parent="a").to_interval(
            axis="x"
        ) == LayoutBlock(q.to_interval(axis="x"), id=1, type=2, parent="a")
        assert LayoutBlock(q, id=1, type=2, parent="a").to_interval(
            axis="y"
        ) == LayoutBlock(q.to_interval(axis="y"), id=1, type=2, parent="a")
        assert LayoutBlock(
            q, id=1, type=2, parent="a"
        ).to_rectangle() == LayoutBlock(
            q.to_rectangle(), id=1, type=2, parent="a"
        )
        assert LayoutBlock(
            q, id=1, type=2, parent="a"
        ).to_quadrilateral() == LayoutBlock(q, id=1, type=2, parent="a")

        with pytest.raises(ValueError):
            LayoutBlock(q, id=1, type=2, parent="a").to_interval()
            LayoutBlock(r, id=1, type=2, parent="a").to_interval()

    def test_layout(self):  # pylint: disable=too-many-statements
        i = Interval(4, 5, axis="y")
        q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
        r = Rectangle(3, 3, 5, 6)
        t = LayoutBlock(i, id=1, type=2, text="12")

        # Test Initializations
        l = Layout([i, q, r])
        l = Layout((i, q))
        Layout([l])
        with pytest.raises(ValueError):
            Layout(l)

        # Test tuple-like inputs
        l = Layout((i, q, r))
        assert l._blocks == [i, q, r]  # pylint: disable=protected-access
        l.append(i)

        # Test apply functions
        l = Layout([i, q, r])
        l.get_texts()
        assert l.filter_by(t) == Layout([i])
        assert l.condition_on(i) == Layout(
            [block.condition_on(i) for block in [i, q, r]]
        )
        assert l.relative_to(q) == Layout(
            [block.relative_to(q) for block in [i, q, r]]
        )
        assert l.is_in(r) == Layout([block.is_in(r) for block in [i, q, r]])
        assert l.get_homogeneous_blocks() == [
            i.to_quadrilateral(),
            q,
            r.to_quadrilateral(),
        ]

        i2 = LayoutBlock(i, id=1, type=2, text="12")
        r2 = LayoutBlock(r, id=1, type=2, parent="a")
        q2 = LayoutBlock(q, id=1, type=2, next="a")
        l2 = Layout([i2, r2, q2], page_data={"width": 200, "height": 200})

        l2.get_texts()
        l2.get_info("next")
        l2.condition_on(i)
        l2.relative_to(q)
        l2.filter_by(t)
        l2.is_in(r)

        l2.scale(4)
        l2.shift(4)
        l2.pad(left=2)

        # Test slicing function
        homogeneous_blocks = l2[:2].get_homogeneous_blocks()
        assert homogeneous_blocks[0].block == i.to_rectangle()
        assert homogeneous_blocks[1].block == r

        # Test appending and extending
        assert l + [i2] == Layout([i, q, r, i2])
        assert l + l == Layout([i, q, r] * 2)
        l.append(i)
        assert l == Layout([i, q, r, i])
        l2.extend([q])
        assert l2 == Layout(
            [i2, r2, q2, q], page_data={"width": 200, "height": 200}
        )

        # Test sort
        ## When sorting inplace, it should return None
        l = Layout([i])
        assert l.sort(key=lambda x: x.coordinates[1], inplace=True) is None

        ## Make sure only sorting inplace works
        l = Layout([i, i.shift(2)])
        l.sort(key=lambda x: x.coordinates[1], reverse=True)
        assert l != Layout([i.shift(2), i])
        l.sort(key=lambda x: x.coordinates[1], reverse=True, inplace=True)
        assert l == Layout([i.shift(2), i])

        l = Layout([q, r, i], page_data={"width": 200, "height": 400})
        assert l.sort(key=lambda x: x.coordinates[0]) == Layout(
            [i, q, r], page_data={"width": 200, "height": 400}
        )

        l = Layout([q, t])
        assert l.sort(key=lambda x: x.coordinates[0]) == Layout([t, q])

    def test_layout_comp(self):
        a = Layout([Rectangle(1, 2, 3, 4)])
        b = Layout([Rectangle(1, 2, 3, 4)])

        assert a == b

        a.append(Rectangle(1, 2, 3, 5))
        assert a != b
        b.append(Rectangle(1, 2, 3, 5))
        assert a == b

        a = Layout([LayoutBlock(Rectangle(1, 2, 3, 4))])
        assert a != b

    def test_shape_operations(self):
        i_1 = Interval(1, 2, axis="y", canvas_height=30, canvas_width=400)
        i_2 = LayoutBlock(Interval(1, 2, axis="x"))
        i_3 = Interval(1, 2, axis="y")

        r_1 = Rectangle(0.5, 0.5, 2.5, 1.5)
        r_2 = LayoutBlock(Rectangle(0.5, 0.5, 2, 2.5))

        q_1 = Quadrilateral([[1, 1], [2.5, 1.2], [2.5, 3], [1.5, 3]])
        q_2 = LayoutBlock(
            Quadrilateral([[0.5, 0.5], [2, 1], [1.5, 2.5], [0.5, 2]])
        )

        # I and I in different axes
        assert i_1.intersect(i_1) == i_1
        assert i_1.intersect(i_2) == Rectangle(1, 1, 2, 2)
        assert (
            i_1.intersect(i_3) == i_1
        )  # Ensure intersect copy the canvas size

        assert i_1.union(i_1) == i_1
        with pytest.raises(InvalidShapeError):
            assert i_1.union(i_2) == Rectangle(1, 1, 2, 2)

        # I and R in different axes
        assert i_1.intersect(r_1) == Rectangle(0.5, 1, 2.5, 1.5)
        assert i_2.intersect(r_1).block == Rectangle(1, 0.5, 2, 1.5)
        assert i_1.union(r_1) == Rectangle(0.5, 0.5, 2.5, 2)
        assert i_2.union(r_1).block == r_1

        # I and Q in strict mode
        with pytest.raises(NotSupportedShapeError):
            i_1.intersect(q_1)
            i_1.union(q_1)

        # I and Q in different axes
        assert i_1.intersect(q_1, strict=False) == Rectangle(1, 1, 2.5, 2)
        assert i_1.union(q_1, strict=False) == Rectangle(1, 1, 2.5, 3)
        assert i_2.intersect(q_1, strict=False).block == Rectangle(1, 1, 2, 3)
        assert i_2.union(q_1, strict=False).block == Rectangle(1, 1, 2.5, 3)

        # R and I
        assert r_1.intersect(i_1) == i_1.intersect(r_1)

        # R and R
        assert (
            r_1.intersect(r_2)
            == r_2.intersect(r_1).block
            == Rectangle(0.5, 0.5, 2, 1.5)
        )
        assert (
            r_1.union(r_2)
            == r_2.union(r_1).block
            == Rectangle(0.5, 0.5, 2.5, 2.5)
        )

        # R and Q
        with pytest.raises(NotSupportedShapeError):
            r_1.intersect(q_1)
            r_1.union(q_1)

        assert r_1.intersect(q_1, strict=False) == Rectangle(1, 1, 2.5, 1.5)
        assert r_1.union(q_1, strict=False) == Rectangle(0.5, 0.5, 2.5, 3)
        assert r_1.intersect(q_2, strict=False) == r_1.intersect(
            q_2.to_rectangle()
        )
        assert r_1.union(q_2, strict=False) == r_1.union(q_2.to_rectangle())

        # Q and others in strict mode
        with pytest.raises(NotSupportedShapeError):
            q_1.intersect(i_1)
            q_1.intersect(r_1)
            q_1.intersect(q_2)

        # Q and I
        assert q_1.intersect(i_1, strict=False) == i_1.intersect(
            q_1, strict=False
        )
        assert q_1.union(i_1, strict=False) == i_1.union(q_1, strict=False)

        # Q and R
        assert q_1.intersect(r_1, strict=False) == r_1.intersect(
            q_1, strict=False
        )
        assert q_1.union(r_1, strict=False) == r_1.union(q_1, strict=False)

        # Q and R
        assert (
            q_1.intersect(q_2, strict=False)
            == q_2.intersect(q_1, strict=False).block
        )
        assert q_1.intersect(q_2, strict=False) == Rectangle(1, 1, 2, 2.5)
        assert (
            q_1.union(q_2, strict=False) == q_2.union(q_1, strict=False).block
        )
        assert q_1.union(q_2, strict=False) == Rectangle(0.5, 0.5, 2.5, 3)

    def test_dict(self):

        i = Interval(1, 2, "y", canvas_height=5)
        i_dict = {
            "block_type": "interval",
            "start": 1,
            "end": 2,
            "axis": "y",
            "canvas_height": 5,
            "canvas_width": 0,
        }
        assert i.to_dict() == i_dict
        assert i == Interval.from_dict(i_dict)

        r = Rectangle(1, 2, 3, 4)
        r_dict = {
            "block_type": "rectangle",
            "x_1": 1,
            "y_1": 2,
            "x_2": 3,
            "y_2": 4,
        }
        assert r.to_dict() == r_dict
        assert r == Rectangle.from_dict(r_dict)

        q = Quadrilateral(np.arange(8).reshape(4, 2), 200, 400)
        q_dict = {
            "block_type": "quadrilateral",
            "points": [0, 1, 2, 3, 4, 5, 6, 7],
            "height": 200,
            "width": 400,
        }
        assert q.to_dict() == q_dict
        assert q == Quadrilateral.from_dict(q_dict)

        l = Layout([i, r, q], page_data={"width": 200, "height": 200})
        l_dict = {
            "page_data": {"width": 200, "height": 200},
            "blocks": [i_dict, r_dict, q_dict],
        }
        assert l.to_dict() == l_dict

        i2 = LayoutBlock(i, "")
        i_dict["text"] = ""
        assert i2.to_dict() == i_dict
        assert i2 == LayoutBlock.from_dict(i_dict)

        r2 = LayoutBlock(r, id=24)
        r_dict["id"] = 24
        assert r2.to_dict() == r_dict
        assert r2 == LayoutBlock.from_dict(r_dict)

        q2 = LayoutBlock(q, text="test", parent=45)
        q_dict["text"] = "test"
        q_dict["parent"] = 45
        assert q2.to_dict() == q_dict
        assert q2 == LayoutBlock.from_dict(q_dict)

        l2 = Layout([i2, r2, q2])
        l2_dict = {"page_data": {}, "blocks": [i_dict, r_dict, q_dict]}
        assert l2.to_dict() == l2_dict

        test_layout_dict = {
            "page_data": {},
            "blocks": [
                {
                    "block_type": "rectangle",
                    "x_1": 55.03478240966797,
                    "y_1": 513.2783813476562,
                    "x_2": 721.6126708984375,
                    "y_2": 796.1692504882812,
                    "type": "Figure",
                    "score": 0.4417479336261749,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 46.30983352661133,
                    "y_1": 405.9814453125,
                    "x_2": 701.2566528320312,
                    "y_2": 532.7406616210938,
                    "type": "Table",
                    "score": 0.8386488556861877,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.54902267456055,
                    "y_1": 994.11474609375,
                    "x_2": 326.46258544921875,
                    "y_2": 1010.3915405273438,
                    "type": "Text",
                    "score": 0.8590222001075745,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.969093322753906,
                    "y_1": 957.962646484375,
                    "x_2": 226.48667907714844,
                    "y_2": 973.9776611328125,
                    "type": "Text",
                    "score": 0.8500787615776062,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 443.3688659667969,
                    "y_1": 168.80490112304688,
                    "x_2": 571.4012451171875,
                    "y_2": 274.23846435546875,
                    "type": "Title",
                    "score": 0.84440678358078,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.391357421875,
                    "y_1": 77.59295654296875,
                    "x_2": 287.9927062988281,
                    "y_2": 127.56832122802734,
                    "type": "Title",
                    "score": 0.8043980598449707,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.621726989746094,
                    "y_1": 912.3032836914062,
                    "x_2": 247.256103515625,
                    "y_2": 933.9706420898438,
                    "type": "Title",
                    "score": 0.7909116148948669,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 56.09678649902344,
                    "y_1": 171.4312286376953,
                    "x_2": 97.61103057861328,
                    "y_2": 192.955322265625,
                    "type": "Title",
                    "score": 0.7554147839546204,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.03081130981445,
                    "y_1": 199.1637420654297,
                    "x_2": 172.71914672851562,
                    "y_2": 257.8761291503906,
                    "type": "Title",
                    "score": 0.6639509797096252,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 299.3883056640625,
                    "y_1": 320.39349365234375,
                    "x_2": 425.7754821777344,
                    "y_2": 373.8882751464844,
                    "type": "Title",
                    "score": 0.6330041885375977,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 54.45736312866211,
                    "y_1": 317.0123291015625,
                    "x_2": 176.74444580078125,
                    "y_2": 374.41131591796875,
                    "type": "Title",
                    "score": 0.5131245255470276,
                },
                {
                    "block_type": "rectangle",
                    "x_1": 55.57542037963867,
                    "y_1": 956.4700317382812,
                    "x_2": 226.97865295410156,
                    "y_2": 973.5662231445312,
                    "type": "Title",
                    "score": 0.49591660499572754,
                },
            ],
        }

        layout = Layout.from_dict(test_layout_dict)
        blocks = layout.get_blocks()
        self.assertEqual(len(blocks), 12)
        self.assertEqual(blocks[0].type, "Figure")
        self.assertEqual(blocks[1].type, "Table")
        self.assertEqual(blocks[2].type, "Text")
        self.assertEqual(layout.to_dict(), test_layout_dict)
