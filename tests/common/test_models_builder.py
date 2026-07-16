import numpy as np
import pytest

from doctr.file_utils import CLASS_NAME
from doctr.io import Document
from doctr.io.elements import KIEDocument, LayoutElement, Table
from doctr.models import builder

words_per_page = 10

boxes_1 = {CLASS_NAME: np.random.rand(words_per_page, 6)}  # dict format
boxes_1[CLASS_NAME][:2] *= boxes_1[CLASS_NAME][2:4]

boxes_2 = np.random.rand(words_per_page, 6)  # array format
boxes_2[:2] *= boxes_2[2:4]


def test_documentbuilder():
    num_pages = 2

    # Don't resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=False, resolve_blocks=False)
    pages = [np.zeros((100, 200, 3))] * num_pages
    boxes = np.random.rand(words_per_page, 6)  # array format
    boxes[:2] *= boxes[2:4]
    objectness_scores = np.array([0.9] * words_per_page)
    # Arg consistency check
    with pytest.raises(ValueError):
        doc_builder(
            pages,
            [boxes, boxes],
            [objectness_scores, objectness_scores],
            [("hello", 1.0)] * 3,
            [(100, 200)] * num_pages,
            [{"value": 0, "confidence": None}] * 3,
        )
    out = doc_builder(
        pages,
        [boxes, boxes],
        [objectness_scores, objectness_scores],
        [[("hello", 1.0)] * words_per_page] * num_pages,
        [(100, 200)] * num_pages,
        [[{"value": 0, "confidence": None}] * words_per_page] * num_pages,
    )
    assert isinstance(out, Document)
    assert len(out.pages) == num_pages
    assert all(isinstance(page.page, np.ndarray) for page in out.pages) and all(
        page.page.shape == (100, 200, 3) for page in out.pages
    )
    # 1 Block & 1 line per page
    assert len(out.pages[0].blocks) == 1 and len(out.pages[0].blocks[0].lines) == 1
    assert len(out.pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = builder.DocumentBuilder(resolve_lines=True, resolve_blocks=True)
    out = doc_builder(
        pages,
        [boxes, boxes],
        [objectness_scores, objectness_scores],
        [[("hello", 1.0)] * words_per_page] * num_pages,
        [(100, 200)] * num_pages,
        [[{"value": 0, "confidence": None}] * words_per_page] * num_pages,
    )

    # No detection
    boxes = np.zeros((0, 4))
    objectness_scores = np.zeros([0])
    out = doc_builder(
        pages,
        [boxes, boxes],
        [objectness_scores, objectness_scores],
        [[], []],
        [(100, 200)] * num_pages,
        [[]] * num_pages,
    )
    assert len(out.pages[0].blocks) == 0

    # Rotated boxes to export as straight boxes
    boxes = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
        [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
    ])
    objectness_scores = np.array([0.99, 0.99])
    doc_builder_2 = builder.DocumentBuilder(resolve_blocks=False, resolve_lines=False, export_as_straight_boxes=True)
    out = doc_builder_2(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        [[("hello", 0.99), ("word", 0.99)]],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 2],
    )
    assert out.pages[0].blocks[0].lines[0].words[-1].geometry == ((0.45, 0.5), (0.6, 0.65))
    assert out.pages[0].blocks[0].lines[0].words[-1].objectness_score == 0.99

    # Repr
    assert (
        repr(doc_builder) == "DocumentBuilder(resolve_lines=True, "
        "resolve_blocks=True, paragraph_break=0.035, export_as_straight_boxes=False, keep_reading_order=False)"
    )


def test_kiedocumentbuilder():
    num_pages = 2

    # Don't resolve lines
    doc_builder = builder.KIEDocumentBuilder(resolve_lines=False, resolve_blocks=False)
    pages = [np.zeros((100, 200, 3))] * num_pages
    predictions = {CLASS_NAME: np.random.rand(words_per_page, 6)}  # dict format
    predictions[CLASS_NAME][:2] *= predictions[CLASS_NAME][2:4]
    objectness_scores = {CLASS_NAME: np.array([0.9] * words_per_page)}
    # Arg consistency check
    with pytest.raises(ValueError):
        doc_builder(
            pages,
            [predictions, predictions],
            [objectness_scores, objectness_scores],
            [{CLASS_NAME: ("hello", 1.0)}] * 3,
            [(100, 200), (100, 200)],
            [{CLASS_NAME: [{"value": 0, "confidence": None}] * 3}],
        )
    out = doc_builder(
        pages,
        [predictions, predictions],
        [objectness_scores, objectness_scores],
        [{CLASS_NAME: [("hello", 1.0)] * words_per_page}] * num_pages,
        [(100, 200), (100, 200)],
        [{CLASS_NAME: [{"value": 0, "confidence": None}] * words_per_page}] * num_pages,
    )
    assert isinstance(out, KIEDocument)
    assert len(out.pages) == num_pages
    assert all(isinstance(page.page, np.ndarray) for page in out.pages) and all(
        page.page.shape == (100, 200, 3) for page in out.pages
    )
    # 1 Block & 1 line per page
    assert len(out.pages[0].predictions) == 1
    assert len(out.pages[0].predictions[CLASS_NAME]) == words_per_page

    # Resolve lines
    doc_builder = builder.KIEDocumentBuilder(resolve_lines=True, resolve_blocks=True)
    out = doc_builder(
        pages,
        [predictions, predictions],
        [objectness_scores, objectness_scores],
        [{CLASS_NAME: [("hello", 1.0)] * words_per_page}] * num_pages,
        [(100, 200), (100, 200)],
        [{CLASS_NAME: [{"value": 0, "confidence": None}] * words_per_page}] * num_pages,
    )

    # No detection
    predictions = {CLASS_NAME: np.zeros((0, 4))}
    objectness_scores = {CLASS_NAME: np.zeros((1))}

    out = doc_builder(
        pages,
        [predictions, predictions],
        [objectness_scores, objectness_scores],
        [{CLASS_NAME: []}, {CLASS_NAME: []}],
        [(100, 200), (100, 200)],
        [{CLASS_NAME: []}, {CLASS_NAME: []}],
    )
    assert len(out.pages[0].predictions[CLASS_NAME]) == 0

    # Rotated boxes to export as straight boxes
    predictions = {
        CLASS_NAME: np.array([
            [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
            [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
        ])
    }
    objectness_scores = {CLASS_NAME: np.array([0.99, 0.99])}
    doc_builder_2 = builder.KIEDocumentBuilder(resolve_blocks=False, resolve_lines=False, export_as_straight_boxes=True)
    out = doc_builder_2(
        [np.zeros((100, 100, 3))],
        [predictions],
        [objectness_scores],
        [{CLASS_NAME: [("hello", 0.99), ("word", 0.99)]}],
        [(100, 100)],
        [{CLASS_NAME: [{"value": 0, "confidence": None}] * 2}],
    )
    assert out.pages[0].predictions[CLASS_NAME][0].geometry == ((0.05, 0.1), (0.2, 0.25))
    assert out.pages[0].predictions[CLASS_NAME][1].geometry == ((0.45, 0.5), (0.6, 0.65))
    assert out.pages[0].predictions[CLASS_NAME][1].objectness_score == 0.99

    # Repr
    assert (
        repr(doc_builder) == "KIEDocumentBuilder(resolve_lines=True, "
        "resolve_blocks=True, paragraph_break=0.035, export_as_straight_boxes=False, keep_reading_order=False)"
    )


def test_documentbuilder_layout():

    doc_builder = builder.DocumentBuilder()
    boxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
    objectness_scores = np.array([0.9, 0.9])
    regions = [
        {
            "boxes": np.array([[0.05, 0.02, 0.95, 0.08], [0.05, 0.2, 0.95, 0.5]], dtype=np.float32),
            "class_names": ["Title", "Text"],
            "scores": [0.95, 0.88],
        }
    ]
    out = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        [[("hello", 0.99), ("world", 0.99)]],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 2],
        regions=regions,
    )
    page = out.pages[0]
    # Layout regions are attached as LayoutElement and exported
    assert len(page.layout) == 2
    assert all(isinstance(region, LayoutElement) for region in page.layout)
    assert [region.type for region in page.layout] == ["Title", "Text"]
    assert page.layout[0].confidence == pytest.approx(0.95)
    assert np.allclose(np.array(page.layout[0].geometry), [[0.05, 0.02], [0.95, 0.08]], atol=1e-6)
    assert page.export()["layout"] == [region.export() for region in page.layout]

    # no regions -> empty layout
    out_no_layout = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        [[("hello", 0.99), ("world", 0.99)]],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 2],
    )
    assert out_no_layout.pages[0].layout == []
    assert out_no_layout.pages[0].export()["layout"] == []

    # Rotated layout polygons (4, 2) are converted to a 4-point geometry
    rotated_regions = [
        {
            "boxes": np.array([[[0.1, 0.1], [0.4, 0.12], [0.39, 0.3], [0.09, 0.28]]], dtype=np.float32),
            "class_names": ["Table"],
            "scores": [0.7],
        }
    ]
    out_rot = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        [[("hello", 0.99), ("world", 0.99)]],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 2],
        regions=rotated_regions,
    )
    region = out_rot.pages[0].layout[0]
    assert region.type == "Table"
    assert isinstance(region.geometry, tuple) and len(region.geometry) == 4


def _table_cell(x0, y0, x1, y1, rs, re, cs, ce, score=0.9):
    return {
        "geometry": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "score": score,
        "row_start": rs,
        "row_end": re,
        "col_start": cs,
        "col_end": ce,
    }


def test_documentbuilder_tables():
    doc_builder = builder.DocumentBuilder(resolve_lines=True)

    # 4 words inside a top table, 2 inside a bottom table, 1 caption outside both
    def wbox(cx, cy, w=0.04, h=0.02):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    words = [
        ("Name", 0.17, 0.15),
        ("Age", 0.36, 0.15),
        ("Alice", 0.17, 0.27),
        ("30", 0.36, 0.27),
        ("City", 0.17, 0.63),
        ("Pop", 0.36, 0.63),
        ("caption", 0.30, 0.92),
    ]
    boxes = np.array([wbox(cx, cy) for _, cx, cy in words], dtype=np.float32)
    text_preds = [[(w, 0.95) for w, _, _ in words]]
    objectness_scores = np.full(len(words), 0.9, dtype=np.float32)
    orientations = [[{"value": 0, "confidence": None}] * len(words)]

    # The OCR pipeline passes a list of grids (one per cropped table region), in page-relative coordinates.
    # The bottom table uses offset (1-based) logical coordinates to exercise local re-indexing.
    table_top = {
        "cells": [
            _table_cell(0.10, 0.10, 0.25, 0.20, 0, 0, 0, 0),
            _table_cell(0.28, 0.10, 0.45, 0.20, 0, 0, 1, 1),
            _table_cell(0.10, 0.22, 0.25, 0.32, 1, 1, 0, 0),
            _table_cell(0.28, 0.22, 0.45, 0.32, 1, 1, 1, 1),
        ],
        "num_rows": 2,
        "num_cols": 2,
    }
    table_bottom = {
        "cells": [
            _table_cell(0.10, 0.58, 0.25, 0.68, 1, 1, 1, 1),
            _table_cell(0.28, 0.58, 0.45, 0.68, 1, 1, 2, 2),
        ],
        "num_rows": 99,  # deliberately wrong dims -> recomputed from local coordinates
        "num_cols": 99,
    }

    out = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        text_preds,
        [(100, 100)],
        orientations,
        tables=[[table_top, table_bottom]],
    )
    page = out.pages[0]

    # One Table per provided grid
    assert len(page.tables) == 2
    assert all(isinstance(t, Table) for t in page.tables)
    assert page.tables[0].to_grid() == [["Name", "Age"], ["Alice", "30"]]
    # bottom table re-indexed from offset coordinates to a local 0-based 1 x 2 grid
    assert (page.tables[1].num_rows, page.tables[1].num_cols) == (1, 2)
    assert page.tables[1].to_grid() == [["City", "Pop"]]

    # Words assigned to a table are removed from the blocks; the caption remains
    remaining = [w.value for b in page.blocks for line in b.lines for w in line.words]
    assert remaining == ["caption"]

    # Tables are part of the page export
    exported = page.export()
    assert len(exported["tables"]) == 2
    assert page.tables[0].to_grid() == [["Name", "Age"], ["Alice", "30"]]

    # A single grid (dict) is also accepted -> one table
    out_single = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes[:4]],
        [objectness_scores[:4]],
        [text_preds[0][:4]],
        [(100, 100)],
        [orientations[0][:4]],
        tables=[table_top],
    )
    assert len(out_single.pages[0].tables) == 1

    # No tables -> empty page.tables and every word is kept in the blocks
    out_none = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        text_preds,
        [(100, 100)],
        orientations,
    )
    assert out_none.pages[0].tables == []
    assert out_none.pages[0].export()["tables"] == []
    kept = sorted(w.value for b in out_none.pages[0].blocks for line in b.lines for w in line.words)
    assert kept == sorted(w for w, _, _ in words)


def test_kiedocumentbuilder_layout():
    from doctr.io.elements import LayoutElement

    doc_builder = builder.KIEDocumentBuilder()
    predictions = {CLASS_NAME: np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])}
    objectness_scores = {CLASS_NAME: np.array([0.9, 0.9])}
    regions = [
        {
            "boxes": np.array([[0.05, 0.02, 0.95, 0.08], [0.05, 0.2, 0.95, 0.5]], dtype=np.float32),
            "class_names": ["Title", "Text"],
            "scores": [0.95, 0.88],
        }
    ]
    out = doc_builder(
        [np.zeros((100, 100, 3))],
        [predictions],
        [objectness_scores],
        [{CLASS_NAME: [("hello", 0.99), ("world", 0.99)]}],
        [(100, 100)],
        [{CLASS_NAME: [{"value": 0, "confidence": None}] * 2}],
        regions=regions,
    )
    page = out.pages[0]
    assert len(page.layout) == 2
    assert all(isinstance(region, LayoutElement) for region in page.layout)
    assert [region.type for region in page.layout] == ["Title", "Text"]
    assert page.export()["layout"] == [region.export() for region in page.layout]

    # no regions -> empty layout
    out_no_layout = doc_builder(
        [np.zeros((100, 100, 3))],
        [predictions],
        [objectness_scores],
        [{CLASS_NAME: [("hello", 0.99), ("world", 0.99)]}],
        [(100, 100)],
        [{CLASS_NAME: [{"value": 0, "confidence": None}] * 2}],
    )
    assert out_no_layout.pages[0].layout == []


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.49, 0.35, 0.59], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [
            [
                [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
                [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
            ],
            [0, 1],
        ],  # rot
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):
    doc_builder = builder.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes))[0].tolist() == sorted_idxs


@pytest.mark.parametrize(
    "input_boxes, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.14, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.18, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.48, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [[0], [1], [2]]],  # 2 lines
        [
            [
                [[0.1, 0.1], [0.2, 0.2], [0.15, 0.25], [0.05, 0.15]],
                [[0.5, 0.5], [0.6, 0.6], [0.55, 0.65], [0.45, 0.55]],
            ],
            [[0], [1]],
        ],  # rot
    ],
)
def test_resolve_lines(input_boxes, lines):
    doc_builder = builder.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes)) == lines


def test_points_in_polygons():
    polys = np.array(
        [
            [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],  # axis-aligned quad
            [[0.5, 0.5], [0.8, 0.6], [0.7, 0.9], [0.45, 0.8]],  # rotated quad
        ],
        dtype=np.float32,
    )
    points = np.array([[0.2, 0.2], [0.6, 0.7], [0.95, 0.95], [0.05, 0.05]], dtype=np.float32)
    inside = builder.DocumentBuilder._points_in_polygons(points, polys)
    assert inside.shape == (4, 2)
    assert inside[0].tolist() == [True, False]
    assert inside[1].tolist() == [False, True]
    assert not inside[2].any()
    assert not inside[3].any()
    # empty inputs yield empty masks instead of raising
    assert builder.DocumentBuilder._points_in_polygons(np.zeros((0, 2)), polys).shape == (0, 2)
    assert builder.DocumentBuilder._points_in_polygons(points, np.zeros((0, 4, 2))).shape == (4, 0)


def test_as_cell_polygon():
    # flat straight box (xmin, ymin, xmax, ymax) -> (4, 2) polygon
    poly = builder.DocumentBuilder._as_cell_polygon([0.1, 0.2, 0.5, 0.6])
    assert poly.shape == (4, 2)
    assert np.allclose(poly, [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]])
    # (4, 2) polygons pass through unchanged
    quad = np.array([[0.1, 0.2], [0.5, 0.25], [0.45, 0.6], [0.05, 0.55]], dtype=np.float32)
    assert np.allclose(builder.DocumentBuilder._as_cell_polygon(quad), quad)


def test_documentbuilder_tables_straight_geometry():
    # Straight-mode table predictions store cells as flat (xmin, ymin, xmax, ymax) boxes.
    # Regression test: this format used to crash the word-to-cell assignment with an IndexError.
    doc_builder = builder.DocumentBuilder()

    def wbox(cx, cy, w=0.04, h=0.02):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    words = [("Name", 0.17, 0.15), ("Age", 0.36, 0.15), ("caption", 0.30, 0.92)]
    boxes = np.array([wbox(cx, cy) for _, cx, cy in words], dtype=np.float32)
    text_preds = [[(w, 0.95) for w, _, _ in words]]
    objectness_scores = np.full(len(words), 0.9, dtype=np.float32)
    orientations = [[{"value": 0, "confidence": None}] * len(words)]

    def straight_cell(x0, y0, x1, y1, cs):
        return {
            "geometry": [x0, y0, x1, y1],  # flat straight-box format, as emitted by the table predictor
            "score": 0.9,
            "row_start": 0,
            "row_end": 0,
            "col_start": cs,
            "col_end": cs,
        }

    table = {
        "cells": [straight_cell(0.10, 0.10, 0.25, 0.20, 0), straight_cell(0.28, 0.10, 0.45, 0.20, 1)],
        "score": 0.9,
    }

    out = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [objectness_scores],
        text_preds,
        [(100, 100)],
        orientations,
        tables=[[table]],
    )
    page = out.pages[0]

    assert len(page.tables) == 1
    assert page.tables[0].to_grid() == [["Name", "Age"]]
    # words assigned to the table are removed from the blocks; the caption remains
    remaining = [w.value for b in page.blocks for line in b.lines for w in line.words]
    assert remaining == ["caption"]


def test_documentbuilder_argument_length_validation():
    # A single mismatched argument must raise, whichever one it is
    num_pages = 2
    doc_builder = builder.DocumentBuilder()
    pages = [np.zeros((100, 200, 3))] * num_pages
    boxes = np.random.rand(words_per_page, 6)
    boxes[:2] *= boxes[2:4]
    page_boxes = [boxes[:, :4]] * num_pages
    objectness = [np.array([0.9] * words_per_page)] * num_pages
    text_preds = [[("hello", 0.99)] * words_per_page] * num_pages
    page_shapes = [(100, 200)] * num_pages
    crop_orientations = [[{"value": 0, "confidence": None}] * words_per_page] * num_pages

    args = dict(
        pages=pages,
        boxes=page_boxes,
        objectness_scores=objectness,
        text_preds=text_preds,
        page_shapes=page_shapes,
        crop_orientations=crop_orientations,
    )
    # sanity: consistent arguments build fine
    assert isinstance(doc_builder(**args), Document)

    # each argument mismatched on its own must raise
    for key in ("pages", "objectness_scores", "text_preds", "page_shapes", "crop_orientations"):
        bad_args = dict(args)
        bad_args[key] = args[key] + [args[key][0]]
        with pytest.raises(ValueError):
            doc_builder(**bad_args)


def test_sort_boxes_degenerate_heights():
    # Boxes with zero height must not produce a NaN ordering (division by median height)
    doc_builder = builder.DocumentBuilder()
    boxes = np.array([[0.5, 0.2, 0.6, 0.2], [0.1, 0.2, 0.2, 0.2]], dtype=np.float32)
    idxs, _ = doc_builder._sort_boxes(boxes)
    assert sorted(np.asarray(idxs).tolist()) == [0, 1]


def test_documentbuilder_tables_empty_cells():
    # A table prediction with no cells (e.g. a false-positive "Table" region where the table model
    # finds nothing) must not crash the document build
    doc_builder = builder.DocumentBuilder()
    boxes = np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
    out = doc_builder(
        [np.zeros((100, 100, 3))],
        [boxes],
        [np.array([0.9])],
        [[("hello", 0.99)]],
        [(100, 100)],
        [[{"value": 0, "confidence": None}]],
        tables=[[{"cells": [], "num_rows": 0, "num_cols": 0}]],
    )
    assert out.pages[0].tables == []
    # the word stays in the regular blocks since it was never consumed by a table
    assert [w.value for b in out.pages[0].blocks for line in b.lines for w in line.words] == ["hello"]


def test_documentbuilder_keep_reading_order():
    # Two columns of 3 lines each: a naive top-down sort interleaves the columns
    left = [[0.1, 0.1 + 0.2 * idx, 0.3, 0.2 + 0.2 * idx] for idx in range(3)]
    right = [[0.6, 0.1 + 0.2 * idx, 0.8, 0.2 + 0.2 * idx] for idx in range(3)]
    boxes = np.asarray(left + right)
    words = [(f"L{idx}", 0.9) for idx in range(3)] + [(f"R{idx}", 0.9) for idx in range(3)]
    crop_orientations = [{"value": 0, "confidence": None}] * len(words)
    args = (
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        [boxes],
        [np.ones(len(words))],
        [words],
        [(100, 100)],
        [crop_orientations],
    )

    doc = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)(*args)
    assert doc.pages[0].render(block_break=" ").split() == ["L0", "L1", "L2", "R0", "R1", "R2"]
    # Without the flag, the blocks keep their original (interleaved) order
    doc = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=False)(*args)
    assert doc.pages[0].render(block_break=" ").split() != ["L0", "L1", "L2", "R0", "R1", "R2"]

    # Layout regions are used to place page furniture: the top line is labeled as a page footer -> emitted last
    boxes = np.asarray([[0.1, 0.05, 0.9, 0.1], [0.1, 0.3, 0.9, 0.4], [0.1, 0.5, 0.9, 0.6]])
    words = [("footer", 0.9), ("first", 0.9), ("second", 0.9)]
    regions = {"boxes": np.asarray([[0.05, 0.02, 0.95, 0.15]]), "class_names": ["Page-footer"], "scores": [0.9]}
    doc = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)(
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        [boxes],
        [np.ones(3)],
        [words],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 3],
        regions=[regions],
    )
    assert doc.pages[0].render(block_break=" ").split() == ["first", "second", "footer"]


def _rot_poly(x0, y0, x1, y1, deg, cx=0.5, cy=0.5):
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return [
        [cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca]
        for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ]


def test_build_tables_rotated_cell_word_order():
    deg = -30
    cb = {
        (0, 0): (0.30, 0.40, 0.62, 0.46),
        (0, 1): (0.63, 0.40, 0.85, 0.46),
        (1, 0): (0.30, 0.47, 0.62, 0.53),
        (1, 1): (0.63, 0.47, 0.85, 0.53),
    }
    cells = [
        {
            "geometry": _rot_poly(*cb[(r, c)], deg),
            "score": 0.9,
            "row_start": r,
            "row_end": r,
            "col_start": c,
            "col_end": c,
        }
        for r in range(2)
        for c in range(2)
    ]
    wb = [
        ("one", 0.31, 0.405, 0.40, 0.455),
        ("two", 0.42, 0.405, 0.51, 0.455),
        ("three", 0.53, 0.405, 0.61, 0.455),
        ("B", 0.64, 0.405, 0.84, 0.455),
        ("C", 0.31, 0.475, 0.61, 0.525),
        ("D", 0.64, 0.475, 0.84, 0.525),
    ]
    polys = np.array([_rot_poly(x0, y0, x1, y1, deg) for _, x0, y0, x1, y1 in wb], dtype=np.float32)
    preds = [(t, 0.9) for t, *_ in wb]
    doc = builder.DocumentBuilder()(
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        [polys],
        [np.ones(6)],
        [preds],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 6],
        tables=[[{"cells": cells, "num_rows": 2, "num_cols": 2}]],
    )
    assert doc.pages[0].tables[0].to_grid()[0][0] == "one two three"


def test_build_tables_nearest_cell_fallback():
    cb = {
        (0, 0): (0.30, 0.40, 0.55, 0.46),
        (0, 1): (0.57, 0.40, 0.82, 0.46),
        (1, 0): (0.30, 0.48, 0.55, 0.54),
        (1, 1): (0.57, 0.48, 0.82, 0.54),
    }
    cells = [
        {"geometry": list(cb[(r, c)]), "score": 0.9, "row_start": r, "row_end": r, "col_start": c, "col_end": c}
        for r in range(2)
        for c in range(2)
    ]
    wb = [
        ("A", 0.31, 0.405, 0.54, 0.455),
        ("B", 0.58, 0.405, 0.81, 0.455),
        ("gap", 0.34, 0.462, 0.52, 0.478),  # center in the gap between the two rows, outside all cells
        ("C", 0.31, 0.485, 0.54, 0.535),
        ("D", 0.58, 0.485, 0.81, 0.535),
        ("BODY", 0.31, 0.70, 0.60, 0.74),
    ]  # far below the table
    polys = np.array([[x0, y0, x1, y1] for _, x0, y0, x1, y1 in wb], dtype=np.float32)
    preds = [(t, 0.9) for t, *_ in wb]
    doc = builder.DocumentBuilder()(
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        [polys],
        [np.ones(6)],
        [preds],
        [(100, 100)],
        [[{"value": 0, "confidence": None}] * 6],
        tables=[[{"cells": cells, "num_rows": 2, "num_cols": 2}]],
    )
    grid = doc.pages[0].tables[0].to_grid()
    assert any("gap" in " ".join(row) for row in grid)
    body = [w.value for b in doc.pages[0].blocks for line in b.lines for w in line.words]
    assert body == ["BODY"]


def test_documentbuilder_keep_reading_order_rotated():
    deg = 25
    height, width = 1000, 800
    angle = np.deg2rad(deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([width / 2, height / 2])

    def _rot_box(x0, y0, x1, y1):
        pts = np.array([
            [x0 * width, y0 * height],
            [x1 * width, y0 * height],
            [x1 * width, y1 * height],
            [x0 * width, y1 * height],
        ])
        return ((pts - center) @ rot.T + center) / [width, height]

    left = [_rot_box(0.1, 0.1 + 0.2 * idx, 0.3, 0.2 + 0.2 * idx) for idx in range(3)]
    right = [_rot_box(0.6, 0.1 + 0.2 * idx, 0.8, 0.2 + 0.2 * idx) for idx in range(3)]
    polys = np.asarray(left + right, dtype=np.float32)
    words = [(f"L{idx}", 0.9) for idx in range(3)] + [(f"R{idx}", 0.9) for idx in range(3)]
    doc = builder.DocumentBuilder(resolve_blocks=True, keep_reading_order=True)(
        [np.zeros((height, width, 3), dtype=np.uint8)],
        [polys],
        [np.ones(len(words))],
        [words],
        [(height, width)],
        [[{"value": 0, "confidence": None}] * len(words)],
    )
    assert doc.pages[0].render(block_break=" ").split() == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_resolve_lines_small_skew_portrait():
    height, width = 4624, 2608

    def _rot_word(x0, y0, x1, y1, deg):
        angle = np.deg2rad(deg)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.array([width / 2, height / 2])
        pts = np.array([
            [x0 * width, y0 * height],
            [x1 * width, y0 * height],
            [x1 * width, y1 * height],
            [x0 * width, y1 * height],
        ])
        return ((pts - center) @ rot.T + center) / [width, height]

    doc_builder = builder.DocumentBuilder()
    for deg in (2, 4, 6, 8):
        polys = np.array(
            [
                _rot_word(0.12 + 0.073 * c, 0.08 + 0.015 * r, 0.185 + 0.073 * c, 0.091 + 0.015 * r, deg)
                for r in range(14)
                for c in range(10)
            ],
            dtype=np.float32,
        )
        lines = doc_builder._resolve_lines(polys, (height, width))
        assert len(lines) == 14, f"{len(lines)} lines at {deg} deg"
        assert all(line == [r * 10 + c for c in range(10)] for r, line in enumerate(lines))
