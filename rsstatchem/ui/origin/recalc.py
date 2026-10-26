"""Recalculation of analysis reports (Origin's lock icon and Recalculate modes), no Qt.

A report sheet produced by a fit or a Statistics dialog stores its operation in
``report.op`` together with a fingerprint of its source data. When a worksheet changes the
main window asks :func:`stale_reports` which reports read from it and whose fingerprint no
longer matches; reports in **Auto** mode are re-run at once (:func:`recalculate`), reports in
**Manual** mode are flagged ``dirty`` (yellow lock) until the user chooses Recalculate, and
**None** never updates.
"""
from __future__ import annotations

import hashlib

from rsstatchem.ui.origin import analysis as an
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import (Project, ReportSheet, Workbook, Worksheet, format_cell,
                                        parse_range, row_window)


def sources(op: dict) -> list[tuple[str, str, str, tuple | None]]:
    """``(book, sheet, column, rows)`` read by an operation."""
    if not op:
        return []
    if op.get("kind") == "fit":
        src = op.get("input") or {}
        rows = tuple(src["rows"]) if src.get("rows") else None
        out = [(src.get("book", ""), src.get("sheet", ""), src.get("y", ""), rows)]
        if src.get("x"):
            out.append((src["book"], src["sheet"], src["x"], rows))
        return out
    out = []
    for text in st.range_strings(op.get("settings") or {}):
        try:
            ref = parse_range(text)
        except ValueError:
            continue
        out.append((ref.book, ref.sheet, ref.col, ref.rows))
    return out


def fingerprint(books: dict, op: dict) -> str:
    """Hash of the source values of an operation (changes when its input data change)."""
    h = hashlib.md5()
    for book, sheet, col, rows in sources(op):
        wb = books.get(book)
        ws = wb.sheet(sheet) if isinstance(wb, Workbook) else None
        if not isinstance(ws, Worksheet):
            h.update(f"missing {book} {sheet}".encode())
            continue
        try:
            c = ws.columns[ws.find(col)]
        except KeyError:
            h.update(f"missing {col}".encode())
            continue
        lo, hi = row_window(ws.n_rows, rows)
        values = (c.values + [None] * ws.n_rows)[lo:hi]
        h.update("\x1f".join(format_cell(v) for v in values).encode("utf-8"))
        h.update(b"\x1e")
        if op.get("kind") == "fit":   # weights come from the Y Error column
            e = ws.yerr_of(ws.find(col))
            if e is not None:
                h.update("\x1f".join(format_cell(v) for v in ws.columns[e].values[lo:hi])
                         .encode("utf-8"))
    return h.hexdigest()


def reports(project: Project) -> list[tuple[Workbook, ReportSheet]]:
    return [(b, s) for b in project.books() for s in b.sheets
            if isinstance(s, ReportSheet) and s.op]


def stale_reports(project: Project, changed: str | None = None) \
        -> list[tuple[Workbook, ReportSheet]]:
    """Reports (not in mode None) reading from book ``changed`` (any when None) whose source
    data differ from the last calculation."""
    books = project.data_windows()
    out = []
    for book, rep in reports(project):
        if rep.mode == "none":
            continue
        srcs = sources(rep.op)
        if changed is not None and not any(s[0] == changed for s in srcs):
            continue
        if fingerprint(books, rep.op) != rep.op.get("hash"):
            out.append((book, rep))
    return out


def stamp(project: Project, report: ReportSheet) -> None:
    """Record the current source fingerprint (after a calculation)."""
    report.op["hash"] = fingerprint(project.data_windows(), report.op)
    report.op["dirty"] = False


def recalculate(project: Project, book: Workbook, report: ReportSheet) -> str:
    """Re-run the operation of ``report`` in place; returns the Results Log text. Raises
    ValueError (AnalysisError / StatError) when the inputs are no longer valid."""
    books = project.data_windows()
    op = report.op
    if op.get("kind") == "fit":
        text = an.refit(book, report, books).log
    elif op.get("kind") == "stats":
        _, text = st.run(op["test"], books, dict(op.get("settings") or {}), report=report)
    else:
        raise ValueError("This report cannot be recalculated.")
    stamp(project, report)
    return "Recalculate: " + text
