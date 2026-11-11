"""Debugging aid: run every Statistics analysis dialog on demo data and report crashes.

    QT_QPA_PLATFORM=offscreen python scripts/analysis_fuzzer.py [--json report.json] [-k text]
                                                               [--start N]

For every dialog in the Data / Transform / Analyze / Graphs / Utilities menus the variable
boxes are filled automatically (by what each box accepts and what its key suggests), then
``validate()`` and ``compute()`` run as OK would. Reported per dialog:

* ``crash``  - compute() or the dialog itself raised something other than a ValueError
               (ValueError is how analyses tell the user about unusable input);
* ``refused`` - a ValueError / validation message (listed so wrong ones can be spotted);
* ``slow``   - compute() took longer than the limit.

A computation stuck inside compiled code cannot be interrupted: after ``LIMIT`` + 30 s the
stacks are dumped to stderr and the process exits with code 3; the index of the dialog is in
``<json>.progress``, so a wrapper can restart with ``--start <index + 1>`` (results are
appended to the JSON report).

Datasets are copies, so procedures that modify data do not affect each other.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RSSTATCHEM_HOME"] = tempfile.mkdtemp(prefix="rsfuzz-")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

LIMIT = 60          # seconds per compute()

GROUP_KEYS = ("group", "factor", "fixed", "by", "strat", "cluster", "row", "col", "layer",
              "break", "class", "block", "panel", "categ", "between", "subject", "id", "split",
              "state", "status", "event", "treat", "arm", "level", "label", "case")
DEPENDENT_KEYS = ("dep", "target", "response", "outcome", "test", "y", "time", "dv", "criterion")


def demo_dataset():
    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(42)
    n = 80
    grp = rng.integers(1, 4, n).astype(float)
    sex = rng.integers(1, 3, n).astype(float)
    x1 = rng.normal(50, 10, n)
    x2 = 0.5 * x1 + rng.normal(0, 6, n)
    x3 = rng.normal(5, 1, n)
    y = 20 + 0.6 * x1 - 0.8 * x3 + 3 * (grp == 2) + rng.normal(0, 5, n)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1.0), "grp": grp, "sex": sex, "x1": x1.round(2),
        "x2": x2.round(2), "x3": x3.round(2), "y": y.round(2),
        "pre": (y - rng.normal(4, 3, n)).round(2), "cnt": rng.poisson(3, n).astype(float),
        "flag": (y > np.median(y)).astype(float),
        "likert1": rng.integers(1, 6, n).astype(float),
        "likert2": rng.integers(1, 6, n).astype(float),
        "likert3": rng.integers(1, 6, n).astype(float),
        "time": rng.exponential(12, n).round(1) + 0.5,
        "txt": rng.choice(["a", "b", "c"], n)})
    df.loc[5, "x2"] = np.nan
    ds = Dataset(df)
    ds.var("grp").value_labels = {1.0: "A", 2.0: "B", 3.0: "C"}
    ds.var("grp").measure = "nominal"
    ds.var("sex").value_labels = {1.0: "L", 2.0: "P"}
    ds.var("sex").measure = "nominal"
    ds.var("flag").value_labels = {0.0: "No", 1.0: "Yes"}
    ds.var("flag").measure = "nominal"
    for name in ("likert1", "likert2", "likert3"):
        ds.var(name).measure = "ordinal"
    return ds


def dialog_entries():
    """(menu path, dialog class) for every dialog class in the Data Editor's menu tree."""
    from rsstatchem.ui.data_editor import MAIN_MENUS, menu_tree

    out = []

    def walk(entries, path):
        for e in entries:
            if not isinstance(e, tuple):
                continue
            text, value = e[0], e[1]
            label = text.replace("&&", "&").rstrip(".")
            if isinstance(value, list):
                walk(value, path + [label])
            elif isinstance(value, type):
                out.append((" > ".join(path + [label]), value))

    tree = menu_tree()
    for top in MAIN_MENUS:
        walk(tree.get(top, []), [top])
    return out


MODE = "plausible"   # or "scale": scale variables everywhere, the classic user mistake


def fill(dlg, ds) -> None:
    """Fill every target box of a VariablePickerDialog with plausible variables."""
    used: set[str] = set()

    def candidates(target, order):
        return [n for n in order if n not in used and dlg._source_item(n) is not None
                and target.accepts(ds.var(n))]

    for target in dlg._targets:
        if target.widget.count():
            continue
        key = target.key.lower()
        if MODE == "scale":
            order = ["x1", "x2", "x3", "y", "pre", "time", "cnt", "likert1", "grp", "sex",
                     "flag", "txt"]
        elif any(k in key for k in GROUP_KEYS) and not any(k == key for k in ("y", "dv")):
            order = ["sex", "grp", "flag", "likert1", "x1", "x2"]
        elif any(key.startswith(k) or k in key for k in DEPENDENT_KEYS):
            order = ["y", "time", "flag", "x1", "x2", "cnt"]
        else:
            order = ["x1", "x2", "x3", "likert1", "likert2", "likert3", "y", "pre", "cnt",
                     "grp", "sex", "flag", "txt"]
        names = candidates(target, order)
        if not names and target.min_items:
            names = candidates(target, list(ds.df.columns))
        take = names[:1] if target.single else names[:3]
        if not target.single and target.min_items > len(take):
            take = names[:target.min_items]
        for name in take:
            if dlg._take(name, target):
                used.add(name)
    for target in dlg._pairs:
        if target.widget.count() == 0:
            try:
                dlg._add_pair(target, "y", "pre")
            except Exception:  # noqa: BLE001 - reported by the caller's validate/compute
                pass
    dlg._update_arrows()
    if hasattr(dlg, "targets_changed"):
        dlg.targets_changed()


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def run_one(path: str, cls, warnings: list) -> dict:
    from rsstatchem.ui.data_editor import _compute_with
    from rsstatchem.ui.dialogs.base import VariablePickerDialog

    ds = demo_dataset()
    result = {"path": path, "dialog": cls.__name__, "status": "ok", "detail": ""}
    dlg = None
    try:
        needs_data = getattr(cls, "needs_data", True)
        dlg = cls(ds, None) if needs_data else cls(None, None)
        if isinstance(dlg, VariablePickerDialog):
            fill(dlg, ds)
            error = dlg.validate()
            if error:
                result.update(status="refused", detail=f"validate: {error}")
                return result
        elif isinstance(dlg, QDialog):
            before = len(warnings)
            dlg.setResult(0)
            dlg.accept()
            if dlg.result() != QDialog.DialogCode.Accepted:
                msg = warnings[before:][-1] if len(warnings) > before else "not accepted"
                result.update(status="refused", detail=f"accept: {msg}")
                return result
        if not hasattr(dlg, "compute"):
            result.update(status="ok", detail="no compute()")
            return result
        t0 = time.perf_counter()
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(LIMIT)
        try:
            out = _compute_with(dlg, ds) if needs_data else dlg.compute()
        finally:
            signal.alarm(0)
        result["seconds"] = round(time.perf_counter() - t0, 2)
        if out is None and not getattr(dlg, "modifies_data", False):
            result.update(status="refused", detail="compute() returned nothing")
    except Timeout:
        result.update(status="slow", detail=f"compute() exceeded {LIMIT} s")
    except ValueError as exc:
        result.update(status="refused", detail=f"ValueError: {exc}")
    except Exception:  # noqa: BLE001
        result.update(status="crash", detail=traceback.format_exc())
    finally:
        if dlg is not None:
            try:
                dlg.close()
                dlg.deleteLater()
            except RuntimeError:
                pass
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json")
    ap.add_argument("-k", help="only dialogs whose menu path contains this text")
    ap.add_argument("--start", type=int, default=0, help="index of the first dialog to run")
    ap.add_argument("--mode", choices=("plausible", "scale"), default="plausible",
                    help="scale: put scale variables in every box, also factor boxes")
    args = ap.parse_args()
    global MODE
    MODE = args.mode

    app = QApplication(sys.argv[:1])
    warnings: list[str] = []

    def box(parent=None, title="", text="", *a, **k):
        warnings.append(f"{title}: {text}")
        return QMessageBox.StandardButton.Ok

    for name in ("information", "warning", "critical"):
        setattr(QMessageBox, name, staticmethod(box))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel)
    QDialog.exec = lambda self, *a, **k: 0

    report = Path(args.json) if args.json else None
    results = json.loads(report.read_text(encoding="utf-8")) \
        if report and report.exists() and args.start else []
    entries = dialog_entries()
    for index, (path, cls) in enumerate(entries):
        if index < args.start or (args.k and args.k.lower() not in path.lower()):
            continue
        if report:
            Path(str(report) + ".progress").write_text(f"{index}\n{path}\n", encoding="utf-8")
        print(f"#{index} {path}", file=sys.stderr, flush=True)
        faulthandler.dump_traceback_later(LIMIT + 30, exit=True)
        r = run_one(path, cls, warnings)
        faulthandler.cancel_dump_traceback_later()
        r["index"] = index
        results.append(r)
        app.processEvents()
        if report:
            report.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                              encoding="utf-8")
        if r["status"] != "ok":
            last = r["detail"].strip().splitlines()[-1] if r["detail"] else ""
            print(f"{r['status']:8} {path}  [{r['dialog']}]  {last[:160]}", flush=True)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{len(results)} dialogs: {counts}")
    if report:
        Path(str(report) + ".progress").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
