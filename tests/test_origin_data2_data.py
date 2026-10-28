"""WP-2 data model and manipulation: the mask API (``Worksheet.masked_rows`` & co.), mask
persistence and row bookkeeping, masked points left out of fits, worksheet operations
against pandas (stack / unstack / pivot / reduce / sort / query / split), Mathematics
(interpolation, average curves, curve math, polygon area, set operations, 2D/3D
interpolation against scipy) and matrix tools."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate as si

from rsstatchem.core.analysis import datamanip as dm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import mathematics as maths
from rsstatchem.core.analysis import matrix as mx
from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin import worksheet_ops as wo
from rsstatchem.ui.origin.model import (Project, Workbook, Worksheet, load_project,
                                        project_from_dict, project_to_dict, save_project)


def _ws(n=10) -> tuple[Workbook, Worksheet]:
    book = Workbook.new("Book1", 2, n)
    ws = book.sheets[0]
    x = np.arange(1.0, n + 1)
    ws.set_column_values(0, list(x))
    ws.set_column_values(1, list(2 * x + 1))
    return book, ws


# ====================================================================== mask API
def test_mask_api_basic_and_point_mask():
    _, ws = _ws()
    assert ws.masked_rows(1) == [] and not ws.has_masks()
    ws.mask_cells("B", [2, 5, 99])                       # out-of-range rows are ignored
    assert ws.masked_rows("B") == [2, 5] and ws.is_masked(5, 1) and ws.has_masks()
    ws.mask_cells(0, [7])
    pm = ws.point_mask([0, 1])
    assert np.flatnonzero(pm).tolist() == [2, 5, 7]
    ws.swap_mask(1, [1, 2])
    assert ws.masked_rows(1) == [1, 5]
    ws.unmask_cells(1, [1])
    ws.clear_mask([0])
    assert ws.masked_rows(1) == [5] and ws.masked_rows(0) == []
    assert ws.mask_where(1, lambda v: v > 18) == 2 and ws.masked_rows(1) == [5, 8, 9]
    ws.clear_mask()
    assert not ws.has_masks()


def test_xy_arrays_excludes_masked_x_or_y_and_row_range():
    _, ws = _ws()
    ws.mask_cells(1, [0])
    ws.mask_cells(0, [3])
    idx, x, y, e = ws.xy_arrays(1)
    assert idx.tolist() == [1, 2, 4, 5, 6, 7, 8, 9] and e is None
    np.testing.assert_allclose(y, 2 * x + 1)
    idx2, *_ = ws.xy_arrays(1, rows=(2, 5))
    assert idx2.tolist() == [1, 2, 4]
    idx3, *_ = ws.xy_arrays(1, honor_mask=False)
    assert len(idx3) == 10
    idx4, x4, *_ = ws.xy_arrays(1, None)                  # against row numbers
    assert x4[0] == 2.0 and len(idx4) == 9


def test_masks_follow_row_insert_delete_permute_and_resize():
    _, ws = _ws()
    ws.mask_cells(1, [2, 6])
    ws.insert_rows(0, 2)
    assert ws.masked_rows(1) == [4, 8]
    ws.delete_rows([0, 5])
    assert ws.masked_rows(1) == [3, 6]
    order = list(range(ws.data_rows()))[::-1]
    n = ws.data_rows()
    v3 = ws.cell(3, 1)
    ws.permute_rows(order, n)
    assert ws.cell(n - 1 - 3, 1) == v3 and ws.is_masked(n - 1 - 3, 1)
    ws.set_rows(4)
    assert all(r < 4 for r in ws.masked_rows(1))


def test_masks_saved_with_the_project_and_old_files_still_load(tmp_path):
    book, ws = _ws()
    ws.mask_cells(1, [1, 4])
    p = Project()
    p.add(book)
    d = project_to_dict(p)
    col = d["windows"][0]["sheets"][0]["columns"][1]
    assert col["mask"] == [1, 4]
    assert "mask" not in d["windows"][0]["sheets"][0]["columns"][0]   # unchanged format
    save_project(p, tmp_path / "m.rsop")
    q, _ = load_project(tmp_path / "m.rsop")
    assert q.windows["Book1"].sheets[0].masked_rows(1) == [1, 4]
    del col["mask"]                                       # a file written before masks
    assert project_from_dict(d).windows["Book1"].sheets[0].masked_rows(1) == []


def test_masked_point_is_excluded_from_a_fit_like_the_subset():
    """WP-1 acceptance check: fitting the unmasked points equals fitting the subset."""
    _, ws = _ws(12)
    y = np.array(ws.columns[1].numeric(12))
    y[4] += 30.0                                          # an outlier
    ws.set_column_values(1, list(y))
    ws.mask_cells(1, [4])
    _, x, yy, _ = ws.xy_arrays(1)
    f = fit.resolve_model("Line")
    masked = fit.fit_curve(x, yy, f)
    keep = np.arange(12) != 4
    subset = fit.fit_curve(np.arange(1.0, 13)[keep], y[keep], f)
    np.testing.assert_allclose(masked.values, subset.values, rtol=1e-12)
    np.testing.assert_allclose(masked.values, [1.0, 2.0], atol=1e-9)


def test_mask_changes_recalc_fingerprint_only_when_masks_exist():
    book, ws = _ws()
    op = {"kind": "fit", "input": {"book": "Book1", "sheet": "Sheet1", "y": "B", "x": "A"}}
    books = {"Book1": book}
    h0 = recalc.fingerprint(books, op)
    ws.mask_cells(1, [3])
    h1 = recalc.fingerprint(books, op)
    ws.clear_mask()
    assert h1 != h0 and recalc.fingerprint(books, op) == h0


# ====================================================================== manipulation
def _long_sheet() -> Worksheet:
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "grp": ["a", "b", "a", "b", "a", "b"],
                       "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    return wo.frame_to_worksheet(df, "Long")


def test_stack_unstack_pivot_match_pandas():
    ws = _long_sheet()
    piv = wo.pivot_table(ws, 0, 1, 2, "sum")
    ref = pd.pivot_table(pd.DataFrame({"id": [1, 1, 2, 2, 3, 3],
                                       "grp": list("ababab"), "val": [1., 2, 3, 4, 5, 6]}),
                         index="id", columns="grp", values="val", aggfunc="sum")
    np.testing.assert_allclose(piv.columns[1].numeric(3), ref["a"].to_numpy())
    np.testing.assert_allclose(piv.columns[2].numeric(3), ref["b"].to_numpy())
    wide = wo.unstack(ws, 2, 1, key_col=0)
    assert [c.comments for c in wide.columns[1:]] == ["grp = a", "grp = b"]
    np.testing.assert_allclose(wide.columns[2].numeric(3), [2, 4, 6])
    st = wo.stack(wide, [1, 2], [0])
    df = pd.DataFrame({"id": [1., 2, 3], "A": [1., 3, 5], "B": [2., 4, 6]})
    ref = df.melt(id_vars=["id"], value_vars=["A", "B"])
    np.testing.assert_allclose(st.columns[2].numeric(6), ref["value"].to_numpy())
    assert st.columns[1].designation == "L"


def test_reduce_rows_dupx_and_evenly_spaced():
    _, ws = _ws(10)
    r = wo.reduce_rows(ws, 3, "mean")
    np.testing.assert_allclose(r.columns[0].numeric(4), [2, 5, 8, 10])
    r1 = wo.reduce_rows(ws, 4, "first", start=1)
    np.testing.assert_allclose(r1.columns[0].numeric(3), [2, 6, 10])
    ws2 = wo.frame_to_worksheet(pd.DataFrame({"x": [1, 1, 2, 2, 3], "y": [1, 3, 5, 7, 9]}))
    d = wo.reduce_duplicate_x(ws2, 0, [1])
    np.testing.assert_allclose(d.columns[1].numeric(3), [2, 6, 9])
    ex = wo.reduce_xy(ws, 0, [1], step=2.5)
    np.testing.assert_allclose(ex.columns[0].numeric(4), [2.25, 4.75, 7.25, 9.75])
    np.testing.assert_allclose(ex.columns[1].numeric(4), [5, 10, 15, 20])


def test_nested_sort_query_split_duplicates_trim():
    ws = _long_sheet()
    ws.mask_cells(2, [0])
    wo.sort_rows(ws, [(1, False), (2, True)])
    assert [ws.cell(r, 1) for r in range(6)] == ["b", "b", "b", "a", "a", "a"]
    assert [ws.cell(r, 2) for r in range(3)] == [2.0, 4.0, 6.0]
    assert ws.masked_rows(2) == [3]                       # moved with the value 1.0
    rows = wo.query_rows(ws, 'col(grp) = "a" and val >= 3')
    assert sorted(ws.cell(r, 2) for r in rows) == [3.0, 5.0]
    rows2 = wo.query_rows(ws, 'col("val") <> 4 && between(id, 1, 2)')
    assert sorted(ws.cell(r, 2) for r in rows2) == [1.0, 2.0, 3.0]
    parts = wo.split(ws, by_col=1)
    assert [p.name for p in parts] == ["Long_b", "Long_a"] and parts[0].data_rows() == 3
    assert len(wo.split(ws, n_rows=4)) == 2
    dup = wo.frame_to_worksheet(pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 6]}))
    assert wo.remove_duplicates(dup).data_rows() == 2
    gaps = wo.frame_to_worksheet(pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]}))
    assert wo.trim_missing(gaps).data_rows() == 1
    with pytest.raises(dm.QueryError):
        wo.query_rows(ws, "__import__('os')")


def test_sort_order_and_paste_transpose_and_mask_condition():
    order = dm.sort_order([[3.0, None, 1.0, "x", 2.0]], [True])
    assert order.tolist() == [2, 4, 0, 3, 1]                 # numbers, text, empty last
    order = dm.sort_order([[3.0, None, 1.0, 2.0]], [False])
    assert order.tolist() == [0, 3, 2, 1]
    _, ws = _ws(4)
    wo.paste_transpose(ws, [["1", "2", "3"], ["4", "5", "6"]], 0, 0)
    assert [ws.cell(r, 0) for r in range(3)] == [1.0, 2.0, 3.0]
    assert [ws.cell(r, 1) for r in range(3)] == [4.0, 5.0, 6.0]
    assert wo.mask_condition(ws, 1, "B > 4") == 3 and ws.masked_rows(1) == [1, 2, 3]


# ====================================================================== mathematics
def test_interpolation_methods_match_scipy():
    x = np.linspace(0, 10, 12)
    y = np.sin(x)
    xn = np.linspace(0, 10, 50)
    np.testing.assert_allclose(maths.interpolate(x, y, xn, "cubic"),
                               si.CubicSpline(x, y)(xn))
    np.testing.assert_allclose(maths.interpolate(x, y, xn, "pchip"),
                               si.PchipInterpolator(x, y)(xn))
    assert np.isnan(maths.interpolate(x, y, [11.0], "linear"))[0]
    assert maths.interpolate(x, y, [11.0], "linear", True)[0] == pytest.approx(
        y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (11 - 10))


def test_average_curves_curve_math_subtract_translate():
    c1 = (np.array([0., 1, 2, 3]), np.array([0., 1, 2, 3]))
    c2 = (np.array([0.5, 1.5, 2.5]), np.array([2., 3, 4]))
    r = maths.average_curves([c1, c2], "union")
    np.testing.assert_allclose(r["x"], [0, 0.5, 1, 1.5, 2, 2.5, 3])
    np.testing.assert_allclose(r["n"], [1, 2, 2, 2, 2, 2, 1])
    np.testing.assert_allclose(r["mean"][2], (1 + 2.5) / 2)
    r2 = maths.average_curves([c1, c2], "overlap", n_points=5)
    np.testing.assert_allclose(r2["x"], np.linspace(0.5, 2.5, 5))
    np.testing.assert_allclose(maths.curve_math(c1[0], c1[1], "-", *c2),
                               [np.nan, 1 - 2.5, 2 - 3.5, np.nan])
    np.testing.assert_allclose(maths.curve_math(c1[0], c1[1], "/", constant=2), c1[1] / 2)
    np.testing.assert_allclose(maths.subtract_line([0, 1, 2], [1, 3, 3]), [0, 1, 0])
    np.testing.assert_allclose(maths.translate([2, 3, 5], to="min"), [0, 1, 3])
    np.testing.assert_allclose(maths.column_math([1, 2], [0, 4], "/"), [np.nan, 0.5])


def test_polygon_area_and_set_operations():
    r = maths.polygon_area([0, 4, 4, 0], [0, 0, 3, 3])
    assert r["area"] == 12 and r["signed_area"] == 12 and r["perimeter"] == 14
    assert (r["cx"], r["cy"]) == (2, 1.5)
    assert maths.polygon_area([0, 0, 4, 4], [0, 3, 3, 0])["signed_area"] == -12
    np.testing.assert_array_equal(maths.set_operation([1, 2, 3], [2, 3, 4], "intersection"),
                                  [2, 3])
    np.testing.assert_array_equal(maths.set_operation([1, 2, 3], [2, 3, 4], "symmetric"),
                                  [1, 4])
    assert list(maths.set_operation(["a", "b"], ["b", 1.0], "union")) == [1.0, "a", "b"]


def test_2d_and_3d_interpolation_against_scipy():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, (300, 3))
    z = 1 + 2 * p[:, 0] - 3 * p[:, 1]
    q = np.array([[0.3, 0.4], [0.6, 0.5]])
    np.testing.assert_allclose(maths.interpolate_2d(p[:, 0], p[:, 1], z, q[:, 0], q[:, 1]),
                               1 + 2 * q[:, 0] - 3 * q[:, 1], atol=1e-10)
    tps = maths.interpolate_2d(p[:, 0], p[:, 1], z, q[:, 0], q[:, 1], "thin_plate_spline")
    ref = si.RBFInterpolator(p[:, :2], z, kernel="thin_plate_spline")(q)
    np.testing.assert_allclose(tps, ref)
    v = p[:, 0] + p[:, 1] - 2 * p[:, 2]
    out = maths.interpolate_3d(*p.T, v, [0.5], [0.4], [0.3])
    assert out[0] == pytest.approx(0.5 + 0.4 - 0.6, abs=1e-10)


def test_matrix_tools():
    a = np.arange(1.0, 5).reshape(2, 2)
    np.testing.assert_allclose(mx.inverse(a) @ a, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(mx.arithmetic(a, a, "*"), a * a)
    assert np.isnan(mx.arithmetic(a, 0, "/")).all()
    with pytest.raises(mx.MatrixError):
        mx.inverse(np.ones((2, 2)))
    z = np.ones((5, 5))
    z[2, 2] = 10
    assert mx.smooth_filter(z, "median", 3)[2, 2] == 1.0
    np.testing.assert_allclose(mx.smooth_filter(np.full((4, 4), 2.0), "gaussian"), 2.0)
    data, mp = mx.crop(np.arange(20.0).reshape(4, 5), (0, 4, 10, 40), 2, 3, 2, 4)
    np.testing.assert_allclose(data, [[6, 7, 8], [11, 12, 13]])
    assert mp == (1.0, 3.0, 20.0, 30.0)
    vol = mx.integrate_volume(np.full((11, 21), 3.0), (0, 2, 0, 1), baseline=1.0)
    assert vol["volume"] == pytest.approx(4.0)
    xs, ys = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 2, 81))
    assert mx.integrate_volume(xs * ys, (0, 1, 0, 2))["volume"] == pytest.approx(1.0)
    r = mx.resample(np.array([[0.0, 1.0], [2.0, 3.0]]), (0, 1, 0, 1), 3, 3)
    assert r[1, 1] == pytest.approx(1.5)
    rot = mx.rotate_any(np.ones((10, 10)), 45)
    assert rot.shape[0] > 10 and np.isnan(rot[0, 0])
    assert mx.statistics(a)["mean"] == 2.5
