# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: decode logic ported from https://github.com/dreamy-xay/TableCenterNet

import math

import numpy as np
from scipy.interpolate import griddata
from shapely import contains_xy, prepare
from shapely.geometry import Polygon

from doctr.models.core import BaseModel
from doctr.utils import order_points

__all__ = ["_TableCenterNet", "TableCenterNetPostProcessor"]


def _get_logic_coords(lc_logic: np.ndarray, col_span: int, row_span: int) -> tuple[int, int, int, int]:
    """Resolve a cell's logical coordinates (start/end column and row) from the per-corner logical
    predictions (`lc_logic` is a (4, 2) array of [col, row] for corners TL, TR, BR, BL) and the cell span.
    Pure numpy port of the reference `get_logic_coords`."""
    col_span = max(1, col_span)
    row_span = max(1, row_span)
    col_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 0]]
    row_lc = [max(1, int(round(float(p)))) for p in lc_logic[:, 1]]
    cols, rows = lc_logic[:, 0], lc_logic[:, 1]

    if col_lc[0] == col_lc[3]:
        start_col = col_lc[0]
        end_col = start_col + col_span - 1
    elif col_lc[1] == col_lc[2]:
        end_col = max(col_span + 1, col_lc[1]) - 1
        start_col = end_col + 1 - col_span
    elif abs(cols[0] - cols[3]) <= abs(cols[1] - cols[2]):
        start_col = max(1, int(round((cols[0] + cols[3]) / 2.0)))
        end_col = start_col + col_span - 1
    else:
        end_col = max(col_span + 1, int(round((cols[1] + cols[2]) / 2.0))) - 1
        start_col = end_col + 1 - col_span

    if row_lc[0] == row_lc[1]:
        start_row = row_lc[0]
        end_row = start_row + row_span - 1
    elif row_lc[2] == row_lc[3]:
        end_row = max(row_span + 1, row_lc[2]) - 1
        start_row = end_row + 1 - row_span
    elif abs(rows[0] - rows[1]) <= abs(rows[2] - rows[3]):
        start_row = max(1, int(round((rows[0] + rows[1]) / 2.0)))
        end_row = start_row + row_span - 1
    else:
        end_row = max(row_span + 1, int(round((rows[2] + rows[3]) / 2.0))) - 1
        start_row = end_row + 1 - row_span

    return start_col, end_col, start_row, end_row


def _bbox_overlap_query(center_polys: np.ndarray, corner_polys: np.ndarray) -> list[np.ndarray]:
    """For each center polygon, the indices of corner polygons whose axis-aligned bounding boxes overlap."""
    c_xmin, c_xmax = center_polys[:, 0::2].min(1), center_polys[:, 0::2].max(1)
    c_ymin, c_ymax = center_polys[:, 1::2].min(1), center_polys[:, 1::2].max(1)
    k_xmin, k_xmax = corner_polys[:, 0::2].min(1), corner_polys[:, 0::2].max(1)
    k_ymin, k_ymax = corner_polys[:, 1::2].min(1), corner_polys[:, 1::2].max(1)
    out = []
    for i in range(center_polys.shape[0]):
        x_ok = (k_xmin <= c_xmax[i]) & (k_xmax >= c_xmin[i])
        y_ok = (k_ymin <= c_ymax[i]) & (k_ymax >= c_ymin[i])
        out.append(np.nonzero(x_ok & y_ok)[0])
    return out


def _table_angle(polys: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Angle of the rows of a table, from its (N, 4, 2) cells.

    The angle is the mean direction of the cell edges modulo 90 degrees, weighted by their length (the direction of a
    short edge is noisy): the corner order of the cells (and a rotation of the table by a multiple of 90 degrees) does
    not matter, the axes are matched in `_logic_from_geometry`.

    Args:
        polys: the cells, of shape (N, 4, 2)
        ref: the cells the angle is estimated from (all of them by default), boolean mask of shape (N,)

    Returns:
        the angle, in radians
    """
    ref_polys = polys if ref is None or not np.any(ref) else polys[ref]
    # The top and left edges of the cells: the 4-fold angle maps both onto the direction of the rows
    edges = np.concatenate([ref_polys[:, 1] - ref_polys[:, 0], ref_polys[:, 3] - ref_polys[:, 0]])
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    weight = np.linalg.norm(edges, axis=1)
    return float(np.arctan2((weight * np.sin(4 * theta)).sum(), (weight * np.cos(4 * theta)).sum()) / 4)


def _rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points of shape (..., 2) by an angle (in radians) around the origin."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.stack([pts[..., 0] * cos_a - pts[..., 1] * sin_a, pts[..., 0] * sin_a + pts[..., 1] * cos_a], axis=-1)


def _deskew_cells(polys: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Rotate (N, 4, 2) cell quadrilaterals around their mean point so the rows and columns of the table become
    horizontal and vertical (cf. `_table_angle`).

    Args:
        polys: the cells, of shape (N, 4, 2)
        ref: the cells the angle is estimated from (all of them by default), boolean mask of shape (N,)

    Returns:
        the deskewed cells, of shape (N, 4, 2), centered on their mean point
    """
    return _rotate_points(polys - polys.reshape(-1, 2).mean(axis=0), -_table_angle(polys, ref))


def _cell_edges(polys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left, right, top and bottom edges of deskewed (N, 4, 2) cells."""
    return polys[..., 0].min(axis=1), polys[..., 0].max(axis=1), polys[..., 1].min(axis=1), polys[..., 1].max(axis=1)


def _suppress_overlapping_cells(polys: np.ndarray, scores: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Indices of the cells to keep (by decreasing score) once the duplicates are removed.

    The cells of a table do not overlap: a cell covering more than `thresh` of a cell with a higher score, or the
    other way around, is a duplicate detection of the same cell.
    """
    left, right, top, bottom = _cell_edges(_deskew_cells(polys))
    areas = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if keep:
            k = np.asarray(keep)
            inter_w = np.minimum(right[i], right[k]) - np.maximum(left[i], left[k])
            inter_h = np.minimum(bottom[i], bottom[k]) - np.maximum(top[i], top[k])
            inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
            if np.any(inter > thresh * np.maximum(np.minimum(areas[i], areas[k]), 1e-9)):
                continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def _cluster_lines(values: np.ndarray, tol: float) -> np.ndarray:
    """Group 1D cell-edge positions into the sorted positions of the grid lines.

    The positions shared by the most edges become lines first, and absorb every edge closer than `tol`: unlike a
    single-linkage grouping, a few stray edges between two lines can not chain them into a single one. An isolated
    edge (shared by no other edge) only makes a line of its own when no line lies within `1.5 * tol`.
    """
    values = np.sort(values)
    half = tol / 2

    def window(lo: np.ndarray | float, hi: np.ndarray | float):
        return np.searchsorted(values, lo, "left"), np.searchsorted(values, hi, "right")

    lo, hi = window(values - half, values + half)
    support = hi - lo  # number of edges around each edge (itself included)
    taken = np.zeros(values.size, dtype=bool)
    lines: list[float] = []
    order = np.argsort(-support, kind="stable")
    # 1st pass: the lines shared by several edges, 2nd pass: the isolated edges
    for shared in (True, False):
        for i in order:
            if taken[i] or (shared and support[i] < 2):
                continue
            if not shared and lines and np.min(np.abs(np.asarray(lines) - values[i])) <= 1.5 * tol:
                taken[i] = True  # a stray edge of a cell, close to an existing line
                continue
            near = values[lo[i] : hi[i]][~taken[lo[i] : hi[i]]]
            center = float(np.median(near))
            start, end = window(center - tol, center + tol)
            taken[start:end] = True
            taken[i] = True
            lines.append(center)
    return np.sort(np.asarray(lines))


def _grid_lines(
    start_edge: np.ndarray, end_edge: np.ndarray, model_logic: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the start / end edges of the cells along one axis into the grid lines of the table.

    Args:
        start_edge: position of the start edge of each cell, shape (N,)
        end_edge: position of the end edge of each cell, shape (N,)
        model_logic: the model's (start, end) logical coordinates of the cells along this axis, shape (N, 2)

    Returns:
        the sorted line positions, and the index of the line each cell starts / ends on (a cell spans at least one
        line interval)
    """
    extent = end_edge - start_edge
    # Tolerance on the alignment of the edges, from the narrowest cells along this axis
    tol = max(0.4 * float(np.percentile(extent, 10)), 1e-6)
    for _ in range(3):
        lines = _cluster_lines(np.concatenate([start_edge, end_edge]), tol)
        start = np.abs(start_edge[:, None] - lines[None]).argmin(axis=1)
        end = np.abs(end_edge[:, None] - lines[None]).argmin(axis=1)
        # The 2 edges of a narrow row / column (less than 10% of the cells) merged into a single line: the
        # tolerance is lowered from the cells it collapsed. A single collapsed cell is a stray detection
        collapsed = (start == end) & (extent > 2.5e-2 * tol)
        if np.count_nonzero(collapsed) < 2:
            break
        new_tol = max(0.4 * float(np.median(extent[collapsed])), 1e-6)
        if new_tol >= 0.9 * tol:
            break
        tol = new_tol
    if lines.size < 2:
        return lines, start, end
    lines, start, end = _merge_split_lines(lines, start, end, start_edge, end_edge, model_logic)
    end = np.clip(end, 1, lines.size - 1)
    start = np.minimum(start, end - 1)
    return lines, start, end


def _merge_split_lines(
    lines: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_edge: np.ndarray,
    end_edge: np.ndarray,
    model_logic: np.ndarray,
    max_shift: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the neighbouring grid lines that are the same line of the table.

    A line of a table photographed in perspective (or slightly curved) drifts along the table, and the clustering of
    its edges splits it in several lines. 2 neighbouring lines are merged when none of their edges moves by more than
    `max_shift` of the extent of its cell, no cell starts and ends on the merged line (the 2 lines of a narrow row /
    column are kept), and the model does not index them differently (e.g. the close but distinct lines of rows /
    columns that only exist in a part of the table).

    Args:
        lines: the sorted line positions
        start: index of the line each cell starts on
        end: index of the line each cell ends on
        start_edge: position of the start edge of each cell
        end_edge: position of the end edge of each cell
        model_logic: the model's (start, end) logical coordinates of the cells along this axis, shape (N, 2)
        max_shift: maximum move of an edge, relative to the extent of its cell

    Returns:
        the merged line positions, and the index of the line each cell starts / ends on
    """
    extent = np.maximum(end_edge - start_edge, 1e-9)
    edges = np.concatenate([start_edge, end_edge])
    edge_extent = np.concatenate([extent, extent])
    # The model index of each edge: a cell starts on the line of its start index, and ends on the line following its
    # last index
    edge_index = np.concatenate([model_logic[:, 0], model_logic[:, 1] + 1])
    lines, start, end = lines.copy(), start.copy(), end.copy()

    def model_index(on: np.ndarray) -> int | None:
        """The index the model gives to a line, when a strict majority of its edges agree on it"""
        values, counts = np.unique(edge_index[on], return_counts=True)
        return int(values[counts.argmax()]) if counts.size and 2 * counts.max() > on.sum() else None

    while lines.size > 2:
        on_line = np.concatenate([start, end])
        best, best_cost = -1, max_shift
        for k in range(lines.size - 1):
            on_pair = (on_line == k) | (on_line == k + 1)
            # a cell between the 2 lines
            if np.any((start == k) & (end == k + 1)):
                continue
            index_k, index_next = model_index(on_line == k), model_index(on_line == k + 1)
            if index_k is not None and index_next is not None and index_k != index_next:
                continue
            merged = float(np.median(edges[on_pair]))
            cost = float(np.max(np.abs(edges[on_pair] - merged) / edge_extent[on_pair]))
            if cost < best_cost:
                best, best_cost = k, cost
        if best < 0:
            break
        on_pair = (on_line == best) | (on_line == best + 1)
        lines[best] = float(np.median(edges[on_pair]))
        lines = np.delete(lines, best + 1)
        start[start > best] -= 1
        end[end > best] -= 1
    return lines, start, end


def _align_table_logic(
    edges: tuple[np.ndarray, ...], logic: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Make the (N, 4) logical coordinates of the cells of a table consistent with their deskewed edges.

    Args:
        edges: the left, right, top and bottom edges of the deskewed cells, each of shape (N,)
        logic: the logical coordinates predicted by the model, of shape (N, 4)
        ref: the cells the grid lines are built from, of shape (N,): the other ones are placed on these lines

    Returns:
        the logical coordinates, of shape (N, 4), whether each cell is kept, of shape (N,), and the left, right, top
        and bottom edges of the cells once the other cells are placed on the grid lines
    """
    left, right, top, bottom = edges
    logic = logic.copy()
    keep = np.ones(logic.shape[0], dtype=bool)
    placed = [left.copy(), right.copy(), top.copy(), bottom.copy()]
    # The table may be rotated by any multiple of 90 degrees (or the corners ordered by the image axes): match each
    # logical axis (columns, rows) with the geometric axis and direction its model coordinates follow
    geo_axes = [(left, right), (top, bottom)]
    corr = np.zeros((2, 2))
    for g, (lo, hi) in enumerate(geo_axes):
        for a, offset in enumerate((0, 2)):
            pos, idx = (lo + hi)[ref], (logic[:, offset] + logic[:, offset + 1])[ref]
            if pos.std() > 0 and idx.std() > 0:
                corr[g, a] = float(np.corrcoef(pos, idx)[0, 1])
    swap = abs(corr[1, 0]) + abs(corr[0, 1]) > abs(corr[0, 0]) + abs(corr[1, 1])
    for a, offset in enumerate((0, 2)):
        g = 1 - a if swap else a
        start_edge, end_edge = geo_axes[g]
        flip = corr[g, a] < 0
        if flip:  # the logical indices grow against the geometric axis
            start_edge, end_edge = -end_edge, -start_edge
        lines, ref_start, ref_end = _grid_lines(start_edge[ref], end_edge[ref], logic[ref, offset : offset + 2])
        if lines.size < 2:  # degenerate (flat) cells: nothing to align
            continue
        # The other cells are placed on the nearest lines: the ones spanning no line interval are dropped
        start = np.abs(start_edge[:, None] - lines[None]).argmin(axis=1)
        end = np.abs(end_edge[:, None] - lines[None]).argmin(axis=1)
        start[ref], end[ref] = ref_start, ref_end
        keep &= end > start
        end = np.maximum(end, start + 1)
        # and their edges are the lines they are placed on
        last = np.minimum(end, lines.size - 1)  # a dropped cell may start on the last line
        lo, hi = (-lines[last], -lines[start]) if flip else (lines[start], lines[last])
        placed[2 * g][~ref], placed[2 * g + 1][~ref] = lo[~ref], hi[~ref]
        # Each grid line is a logical index: a cell spans the indices between the lines it starts and ends on. The
        # cells of a visual row / column share their indices even when the model drifts on some of them, and cells
        # that do not overlap do not share a logical position
        new_start, new_end = start, end - 1
        # The model indices count the rows / columns before the first line (e.g. an undetected header row): their
        # offset is kept when most of the cells agree on it
        diff = (logic[:, offset] - new_start)[ref]
        values, counts = np.unique(diff, return_counts=True)
        shift = max(int(values[counts.argmax()]), 0) if 2 * counts.max() > diff.size else 0
        logic[:, offset] = new_start + shift
        logic[:, offset + 1] = new_end + shift
    return logic, keep, placed


def _logic_from_geometry(
    polys: np.ndarray, model_logic: np.ndarray, anchored: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make the logical coordinates `[start_col, end_col, start_row, end_row]` (0-indexed) of (N, 4, 2) cells
    consistent with their geometry.

    The cell edges are grouped into the row and column lines of the grid, and each line becomes a logical index: the
    cells of a visual row / column share their indices, even when the logical-coordinate head of the model drifts on
    some of them (borderless tables, long tables), and cells that do not overlap do not share a logical position. The
    model coordinates give the direction of the axes (the table may be rotated) and the offset of the first row /
    column.

    Args:
        polys: the cells, of shape (N, 4, 2)
        model_logic: the logical coordinates predicted by the model, of shape (N, 4)
        anchored: whether the geometry of each cell is reliable (e.g. its corners were relocated on detected corners),
            of shape (N,). The grid lines are built from these cells, the other ones are placed on them (their edges
            become the lines they are placed on), or dropped when they span no row / column. All the cells are
            reliable by default.

    Returns:
        the logical coordinates, of shape (N, 4), whether each cell is kept, of shape (N,), and the cells, of shape
        (N, 4, 2), the unreliable ones placed on the grid lines
    """
    anchored = np.ones(polys.shape[0], dtype=bool) if anchored is None else np.asarray(anchored, dtype=bool)
    # A cell whose geometry is not reliable (e.g. a partial cell cut by the image border) has drifting edges, and
    # would add lines
    ref = anchored if np.count_nonzero(anchored) >= 2 else np.ones_like(anchored)
    angle, center = _table_angle(polys, ref), polys.reshape(-1, 2).mean(axis=0)
    deskewed = _rotate_points(polys - center, -angle)
    logic, keep, (left, right, top, bottom) = _align_table_logic(
        _cell_edges(deskewed), np.asarray(model_logic, dtype=np.int64), ref
    )
    placed = polys.copy()
    if not ref.all():
        rects = np.stack([np.stack([left, top], -1), np.stack([right, top], -1), np.stack([right, bottom], -1)], 1)
        rects = np.concatenate([rects, np.stack([left, bottom], -1)[:, None]], axis=1)
        placed[~ref] = (_rotate_points(rects[~ref], angle) + center).astype(polys.dtype)
    return logic, keep, placed


def _free_logical_cells(logic: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Mask of the cells to keep so that no 2 cells share a logical position: the cells are taken by decreasing
    score, and a cell spanning a position already taken is dropped (e.g. a cell partly overlapping 2 cells).

    Args:
        logic: the logical coordinates `[start_col, end_col, start_row, end_row]` of the cells, of shape (N, 4)
        scores: the scores of the cells, of shape (N,)

    Returns:
        boolean mask of shape (N,)
    """
    keep = np.zeros(logic.shape[0], dtype=bool)
    if logic.shape[0] == 0:
        return keep
    lo = logic.min(axis=0)
    taken = np.zeros((logic[:, 3].max() - lo[2] + 1, logic[:, 1].max() - lo[0] + 1), dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        c0, c1, r0, r1 = logic[i] - lo[[0, 0, 2, 2]]
        if not taken[r0 : r1 + 1, c0 : c1 + 1].any():
            taken[r0 : r1 + 1, c0 : c1 + 1] = True
            keep[i] = True
    return keep


def _edge_ratio(quad: np.ndarray) -> float:
    """Largest length ratio between the opposite edges of a (4, 2) quadrilateral (1 for a parallelogram)"""
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)  # top, right, bottom, left
    lengths = np.maximum(lengths, 1e-6)
    return float(
        max(lengths[0] / lengths[2], lengths[2] / lengths[0], lengths[1] / lengths[3], lengths[3] / lengths[1])
    )


def _misplaced_corners(origin: np.ndarray, cell: np.ndarray, max_ratio: float = 1.5) -> list[int]:
    """Corners of a cell relocated on the corner of another cell, to be moved back to their regressed position.

    The regressed quadrilateral of a cell is roughly a parallelogram (a rectangle in perspective). A corner relocated
    on the corner of a neighbouring cell (e.g. a row further) makes 2 opposite edges of very different lengths: the
    relocated corners are moved back, the one restoring the shape the most first, until the shape is restored.

    Args:
        origin: the regressed quadrilateral of the cell, of shape (4, 2)
        cell: the quadrilateral after the corner relocation, of shape (4, 2)
        max_ratio: maximum length ratio of 2 opposite edges (when the regressed quadrilateral is within it)

    Returns:
        the indices of the corners to move back
    """
    limit = max(max_ratio, _edge_ratio(origin))
    cell = cell.copy()
    moved = [k for k in range(4) if not np.array_equal(cell[k], origin[k])]
    reverted: list[int] = []
    while moved and _edge_ratio(cell) > limit:
        ratios = []
        for k in moved:
            candidate = cell.copy()
            candidate[k] = origin[k]
            ratios.append(_edge_ratio(candidate))
        k = moved.pop(int(np.argmin(ratios)))
        cell[k] = origin[k]
        reverted.append(k)
    return reverted


def _lookup_logic(lc_map: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample the (2, H, W) logical-coordinate map at a clamped pixel location."""
    h, w = lc_map.shape[1:]
    xi = 0 if x < 0 else (w - 1 if x >= w else int(x))
    yi = 0 if y < 0 else (h - 1 if y >= h else int(y))
    return lc_map[:, yi, xi]


class TableCenterNetPostProcessor:
    """TableCenterNet post-processor turning the model's *decoded* key-points into table cells.

    The cell geometry is returned in **relative** coordinates ([0, 1] w.r.t. the model input), so the
    predictor can undo the pre-processor's padding/resize like the other docTR predictors. When
    `assume_straight_pages=True`, geometries are axis-aligned boxes of shape `(N, 4)`; otherwise they
    are quadrilaterals of shape `(N, 4, 2)`.

    Args:
        center_thresh: minimum score for a cell center to be kept
        corner_thresh: minimum score for a corner to be used during relocation
        not_relocate: if True, skip the corner-relocation step (faster, less accurate)
        assume_straight_pages: whether the pages are assumed to be straight (i.e., no rotation)
        geometric_logic: if True, remove the duplicate (overlapping) cells and derive the logical coordinates from
            the cell geometry: the cell edges are grouped into the grid lines of the table, each line being a logical
            index (the logical-coordinate head drifts on borderless and long tables), and no 2 cells share a logical
            position
        structure_thresh: with `geometric_logic`, minimum score of the cells the grid lines are built from, when they
            make most of the table (the other cells are placed on these lines)
    """

    def __init__(
        self,
        center_thresh: float = 0.2,
        corner_thresh: float = 0.2,
        not_relocate: bool = False,
        assume_straight_pages: bool = True,
        geometric_logic: bool = True,
        structure_thresh: float = 0.3,
    ) -> None:
        self.center_thresh = center_thresh
        self.corner_thresh = corner_thresh
        self.not_relocate = not_relocate
        self.assume_straight_pages = assume_straight_pages
        self.geometric_logic = geometric_logic
        self.structure_thresh = structure_thresh
        # Cell score decay: cells optimised on <= 2 corners get their score scaled.
        self.cell_min_optimize_count = 2
        self.cell_decay_thresh = 0.4

    def _relocate(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b].copy()  # (Kc, 8)
        cs = decoded["center_scores"][b].copy()  # (Kc,)
        spans = decoded["center_spans"][b]  # (Kc, 2)
        corner_polys = decoded["corner_polygons"][b]  # (Kn, 8)
        corner_scores = decoded["corner_scores"][b]  # (Kn,)
        corner_pts = decoded["corner_points"][b]  # (Kn, 2)
        corner_logics = decoded["corner_logics"][b]  # (Kn, 2)
        lc_map = decoded["lc"][b]  # (2, H, W)

        valid_c = np.nonzero(cs >= self.center_thresh)[0]
        valid_k = np.nonzero(corner_scores >= self.corner_thresh)[0]
        queries = (
            _bbox_overlap_query(cp[valid_c], corner_polys[valid_k])
            if valid_k.size
            else [np.array([], int)] * valid_c.size
        )

        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        corner_count = np.zeros(cp.shape[0], dtype=np.int32)
        for qi, i in enumerate(valid_c):
            center_poly = Polygon(cp[i].reshape(4, 2))
            prepare(center_poly)
            cell = cp[i].reshape(4, 2)
            origin = decoded["center_polygons"][b][i].reshape(4, 2)
            lc_logic: list[np.ndarray | None] = [None, None, None, None]
            n_used = n_repeat = 0
            for j in valid_k[queries[qi]]:
                cx, cy = corner_pts[j]
                candidate_pts = corner_polys[j].reshape(4, 2)
                if not contains_xy(center_poly, candidate_pts[:, 0], candidate_pts[:, 1]).any():
                    continue
                # nearest corner index is computed on the ORIGINAL polygon
                idx = int(np.argmin(((origin - [cx, cy]) ** 2).sum(1)))
                ox, oy = origin[idx]
                px, py = cell[idx]
                if px == ox and py == oy:
                    n_used += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
                elif (ox - px) ** 2 + (oy - py) ** 2 >= (ox - cx) ** 2 + (oy - cy) ** 2:
                    n_repeat += 1
                    cell[idx] = [cx, cy]
                    lc_logic[idx] = corner_logics[j]
            reverted = _misplaced_corners(origin, cell)
            for k in reverted:
                cell[k] = origin[k]
                lc_logic[k] = None
            corner_count[i] = n_used + n_repeat - len(reverted)
            for k in range(4):
                if lc_logic[k] is None:
                    lc_logic[k] = _lookup_logic(lc_map, cell[k][0], cell[k][1])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(np.stack(lc_logic), col_span, row_span)  # type: ignore[arg-type]
            cp[i] = cell.reshape(8)

        # Score decay for under-optimised cells, then re-sort
        keep_high = cs >= self.center_thresh
        decay = keep_high & (corner_count <= self.cell_min_optimize_count)
        cs[decay] *= self.cell_decay_thresh
        order = np.argsort(-cs)
        return cp[order], cs[order], logic[order], ~decay[order]

    def _simple(self, decoded: dict[str, np.ndarray], b: int):
        cp = decoded["center_polygons"][b]
        cs = decoded["center_scores"][b]
        spans = decoded["center_spans"][b]
        lc_map = decoded["lc"][b]
        logic = np.zeros((cp.shape[0], 4), dtype=np.int32)
        for i in np.nonzero(cs >= self.center_thresh)[0]:
            cell = cp[i].reshape(4, 2)
            lc_logic = np.stack([_lookup_logic(lc_map, cell[k][0], cell[k][1]) for k in range(4)])
            col_span, row_span = int(round(float(spans[i][0]))), int(round(float(spans[i][1])))
            logic[i] = _get_logic_coords(lc_logic, col_span, row_span)
        # No corner relocation: every cell is taken as is
        return cp, cs, logic, np.ones(cs.shape[0], dtype=bool)

    def __call__(self, decoded: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        feat_h, feat_w = decoded["feat_size"]
        scale = np.array([feat_w, feat_h], dtype=np.float32)
        results: list[dict[str, np.ndarray]] = []
        for b in range(decoded["center_polygons"].shape[0]):
            cp, cs, logic, anchored = self._simple(decoded, b) if self.not_relocate else self._relocate(decoded, b)
            keep = cs >= self.center_thresh
            polys = cp[keep].reshape(-1, 4, 2) / scale  # relative coordinates
            polys = np.clip(polys.astype(np.float32), 0, 1)
            scores = cs[keep].astype(np.float32)
            logic = (logic[keep] - 1).astype(np.int32)  # start_col, end_col, start_row, end_row (0-indexed)
            if self.geometric_logic and polys.shape[0] > 0:
                kept = _suppress_overlapping_cells(polys, scores)
                polys, scores, anchored = polys[kept], scores[kept], anchored[keep][kept]
                # The grid lines are built from the confident cells when they make most of the table
                confident = anchored & (scores >= self.structure_thresh)
                logic, valid, polys = _logic_from_geometry(
                    polys, logic[kept], confident if 2 * np.count_nonzero(confident) > scores.size else anchored
                )
                polys = np.clip(polys, 0, 1)
                polys, scores, logic = polys[valid], scores[valid], logic[valid].astype(np.int32)
                free = _free_logical_cells(logic, scores)
                polys, scores, logic = polys[free], scores[free], logic[free]
            if self.assume_straight_pages:
                cells = np.concatenate([polys.min(axis=1), polys.max(axis=1)], axis=1).astype(np.float32)
            else:
                cells = (
                    np.stack([order_points(poly) for poly in polys]).astype(np.float32)
                    if polys.shape[0]
                    else polys.reshape(0, 4, 2).astype(np.float32)
                )
            results.append({
                "polygons": cells,  # (N, 4) boxes or (N, 4, 2) quads in relative coordinates
                "scores": scores,
                "logical": logic,  # start_col, end_col, start_row, end_row (0-indexed)
            })
        return results


def _gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    height, width = det_size
    a1, b1, c1 = 1, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2**2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3**2 - 4 * a3 * c3, 0))) / 2
    return min(r1, r2, r3)


def _gaussian_2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = ((s - 1) / 2 for s in shape)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_umich_gaussian(heatmap: np.ndarray, center: np.ndarray, radius: int, k: float = 1.0) -> None:
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _interpolate_polygons(polygons: list[list[tuple]], img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fill each polygon's interior with the linear interpolation of its per-corner value."""
    final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=bool)
    areas = [_polygon_area([(x, y) for x, y, _ in poly]) for poly in polygons]
    for _, poly in sorted(zip(areas, polygons), key=lambda t: t[0]):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, y_min = math.floor(min(xs)), math.floor(min(ys))
        x_max, y_max = math.ceil(max(xs)), math.ceil(max(ys))
        bw, bh = x_max - x_min + 1, y_max - y_min + 1
        if bw <= 0 or bh <= 0:
            continue
        points = np.array([(x - x_min, y - y_min) for x, y, _ in poly], dtype=np.float32)
        values = np.array([v for _, _, v in poly], dtype=np.float32)
        gx, gy = np.meshgrid(np.arange(bw), np.arange(bh))
        grid_points = np.vstack((gx.ravel(), gy.ravel())).T
        try:
            interp = griddata(points, values, grid_points, method="linear", fill_value=-1)
        except Exception:
            continue
        cols = grid_points[:, 0] + x_min
        rows = grid_points[:, 1] + y_min
        valid = (interp >= 0) & (rows >= 0) & (rows < img_size[0]) & (cols >= 0) & (cols < img_size[1])
        rows, cols, vals = rows[valid], cols[valid], interp[valid]
        free = ~mask[rows, cols]
        final_image[rows[free], cols[free]] = vals[free]
        mask[rows[free], cols[free]] = True
    return final_image, mask


def _interpolate_logical_map(cells_logic_coords: list, output_size: tuple[int, int]):
    """Build the dense (2, H, W) logical-coordinate map (column map, row map) and its mask."""
    if not cells_logic_coords:
        return np.zeros((2, *output_size), np.float32), np.zeros((2, *output_size), np.float32)
    cols = [[(x, y, col) for (x, y, col, _row) in cell] for cell in cells_logic_coords]
    rows = [[(x, y, row) for (x, y, _col, row) in cell] for cell in cells_logic_coords]
    col_img, col_mask = _interpolate_polygons(cols, output_size)
    row_img, row_mask = _interpolate_polygons(rows, output_size)
    lc = np.stack([col_img, row_img], axis=0)
    lc_mask = np.stack([col_mask, row_mask], axis=0).astype(np.float32)
    return lc, lc_mask


def _build_table_target(
    cells: np.ndarray,
    logic: np.ndarray,
    output_size: tuple[int, int],
    max_objects: int = 3000,
    max_corners: int = 12000,
) -> dict[str, np.ndarray]:
    """Render the dense TableCenterNet targets (for a single image) consumed by `TableCenterNet.compute_loss`.

    Args:
        cells: (N, 4, 2) cell quadrilaterals (corner order TL, TR, BR, BL) in **output-grid** coordinates
        logic: (N, 4) integer logical coordinates `[start_col, end_col, start_row, end_row]` (0-indexed)
        output_size: (H, W) of the model output grid (input size // down_ratio)
        max_objects: maximum number of cells
        max_corners: maximum number of distinct corners

    Returns:
        the dense target dictionary (numpy arrays) matching the reference schema
    """
    out_h, out_w = output_size
    hm = np.zeros((2, out_h, out_w), np.float32)
    # Center offsets live in the first `max_objects` slots, corner offsets right after them
    reg = np.zeros((max_objects + max_corners, 2), np.float32)
    ct2cn = np.zeros((max_objects, 8), np.float32)
    cn2ct = np.zeros((max_corners, 8), np.float32)
    reg_ind = np.zeros((max_objects + max_corners,), np.int64)
    reg_mask = np.zeros((max_objects + max_corners,), np.float32)
    ct_ind = np.zeros((max_objects,), np.int64)
    ct_mask = np.zeros((max_objects,), np.float32)
    cn_ind = np.zeros((max_corners,), np.int64)
    cn_mask = np.zeros((max_corners,), np.float32)
    ct_cn_ind = np.zeros((max_objects * 4,), np.int64)
    lc_ind = np.zeros((max_objects, 4), np.int64)
    lc_span = np.zeros((max_objects, 2), np.float32)

    corner_dict: dict[str, int] = {}
    drawn_corners: set[str] = set()
    cells_logic_coords: list = []

    # Every cell is drawn on the heat-maps and the logical map, even beyond `max_objects`: otherwise the
    # extra cells are trained as background and the model learns to stop detecting after ~max_objects cells.
    for i in range(len(cells)):
        slotted = i < max_objects  # whether this cell also gets regression targets
        corners = cells[i].reshape(8).astype(np.float32).copy()
        corners[0::2] = np.clip(corners[0::2], 0, out_w - 1)
        corners[1::2] = np.clip(corners[1::2], 0, out_h - 1)
        if len(set(corners[0::2].tolist())) < 2 or len(set(corners[1::2].tolist())) < 2:
            continue  # not an effective quad
        xs, ys = corners[0::2], corners[1::2]
        max_x, min_x, max_y, min_y = xs.max(), xs.min(), ys.max(), ys.min()
        bbox_h, bbox_w = max_y - min_y, max_x - min_x
        if bbox_h <= 0 or bbox_w <= 0:
            continue

        radius = max(0, int(_gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))))
        center = np.array([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0], np.float32)
        ci = center.astype(np.int32)
        flat = ci[1] * out_w + ci[0]
        _draw_umich_gaussian(hm[0], ci, radius)
        if slotted:
            reg[i] = center - ci
            reg_ind[i], reg_mask[i] = flat, 1
            ct_ind[i], ct_mask[i] = flat, 1
            ct2cn[i] = center[[0, 1, 0, 1, 0, 1, 0, 1]] - corners

        start_col, end_col, start_row, end_row = (int(v) + 1 for v in logic[i])
        x1, y1, x2, y2, x3, y3, x4, y4 = corners.tolist()
        clc = [
            (x1, y1, start_col, start_row),
            (x2, y2, end_col + 1, start_row),
            (x3, y3, end_col + 1, end_row + 1),
            (x4, y4, start_col, end_row + 1),
        ]
        cells_logic_coords.append(clc)
        if not slotted:
            # Heat-map only for the corners of the cells beyond `max_objects`
            for j in range(4):
                cint = corners[j * 2 : j * 2 + 2].astype(np.int32)
                key = f"{cint[0]}_{cint[1]}"
                if key not in drawn_corners:
                    drawn_corners.add(key)
                    _draw_umich_gaussian(hm[1], cint, 2)
            continue
        for j, (x, y, _c, _r) in enumerate(clc):
            lc_ind[i, j] = int(y) * out_w + int(x)
        lc_span[i] = (end_col - start_col + 1, end_row - start_row + 1)

        for j in range(4):
            si = j * 2
            corner = corners[si : si + 2]
            cint = corner.astype(np.int32)
            key = f"{cint[0]}_{cint[1]}"
            if key not in corner_dict:
                nc = len(corner_dict)
                if nc >= max_corners:
                    # No regression slot left: still draw the corner so it is not trained as background
                    if key not in drawn_corners:
                        drawn_corners.add(key)
                        _draw_umich_gaussian(hm[1], cint, 2)
                    continue
                corner_dict[key] = nc
                reg[max_objects + nc] = np.abs(corner - cint)
                reg_ind[max_objects + nc] = cint[1] * out_w + cint[0]
                reg_mask[max_objects + nc] = 1
                cn_ind[nc] = cint[1] * out_w + cint[0]
                cn_mask[nc] = 1
                if key not in drawn_corners:
                    drawn_corners.add(key)
                    _draw_umich_gaussian(hm[1], cint, 2)
                cn2ct[nc][si : si + 2] = corner - center
                ct_cn_ind[4 * i + j] = nc * 4 + j
            else:
                idx = corner_dict[key]
                cn2ct[idx][si : si + 2] = corner - center
                ct_cn_ind[4 * i + j] = idx * 4 + j

    lc, lc_mask = _interpolate_logical_map(cells_logic_coords, (out_h, out_w))
    return {
        "hm": hm,
        "reg": reg,
        "reg_ind": reg_ind,
        "reg_mask": reg_mask,
        "ct_ind": ct_ind,
        "ct_mask": ct_mask,
        "cn_ind": cn_ind,
        "cn_mask": cn_mask,
        "ct2cn": ct2cn,
        "cn2ct": cn2ct,
        "ct_cn_ind": ct_cn_ind,
        "lc": lc,
        "lc_mask": lc_mask,
        "lc_ind": lc_ind,
        "lc_span": lc_span,
    }


def _cells_to_polygons(cells: np.ndarray) -> np.ndarray:
    """Convert table cells to quadrilaterals.

    Args:
        cells: relative axis-aligned boxes of shape `(N, 4)` in `(xmin, ymin, xmax, ymax)` format,
            or quadrilaterals of shape `(N, 4, 2)`.

    Returns:
        Relative quadrilaterals of shape `(N, 4, 2)` in TL, TR, BR, BL order.
    """
    if cells.ndim == 3 and cells.shape[1:] == (4, 2):
        return cells
    if cells.ndim == 2 and cells.shape[1:] == (4,):
        xmin, ymin, xmax, ymax = cells.T
        return np.stack(
            [
                np.stack([xmin, ymin], axis=-1),
                np.stack([xmax, ymin], axis=-1),
                np.stack([xmax, ymax], axis=-1),
                np.stack([xmin, ymax], axis=-1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    raise ValueError(f"cells are expected to have shape (N, 4) or (N, 4, 2), got {cells.shape}")


class _TableCenterNet(BaseModel):
    """TableCenterNet for table-structure recognition, as described in the official implementation
    `<https://github.com/dreamy-xay/TableCenterNet>`_.

    This base class holds the framework-agnostic target rendering (`build_target`): the dense maps consumed
    by `compute_loss` are produced here, while `TableCenterNetPostProcessor` decodes the model output.
    """

    max_objects: int = 3000
    max_corners: int = 12000
    assume_straight_pages: bool = False

    def build_target(
        self,
        target: list[dict[str, np.ndarray]],
        output_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Render the dense training targets for a batch from per-image cell annotations.

        Args:
            target: one `{"cells": (N, 4) relative boxes or (N, 4, 2) relative polygons,
                "logic": (N, 4)}` dict per image
            output_shape: (H, W) of the model output grid (input size // down_ratio)

        Returns:
            the batched dense target dictionary (numpy arrays) matching the reference schema
        """
        cells_per_image: list[np.ndarray] = []
        for t in target:
            cells = np.asarray(t["cells"])
            if cells.dtype != np.float32:
                raise AssertionError("the expected dtype of target 'cells' entry is 'np.float32'.")
            cells = _cells_to_polygons(cells)
            if np.any((cells > 1) | (cells < 0)):
                raise ValueError("the 'cells' entry of the target is expected to take values between 0 & 1.")
            cells_per_image.append(cells)

        out_h, out_w = output_shape
        scale = np.array([out_w, out_h], dtype=np.float32)
        per_image = [
            _build_table_target(
                cells * scale,
                np.asarray(t["logic"], dtype=np.int64).reshape(-1, 4),
                (out_h, out_w),
                self.max_objects,
                self.max_corners,
            )
            for t, cells in zip(target, cells_per_image)
        ]
        return {k: np.stack([img[k] for img in per_image], axis=0) for k in per_image[0]}
