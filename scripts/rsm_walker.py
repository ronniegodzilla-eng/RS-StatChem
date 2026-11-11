"""Debugging aid: open RSM / DOE designs of every family and visit every screen.

    QT_QPA_PLATFORM=offscreen python scripts/rsm_walker.py [-k text]

For each tutorial example and for a generated design of each design family (with simulated
responses) the walker shows every navigation node and, on the analysis pages, every tab and
every model-graph view. Exceptions and error boxes are reported with the place they came from.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import menu_crawler as crawler  # noqa: E402  (offscreen platform, patches, event log)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

NAMES = "ABCDEFG"
SPECS = {
    "factorial 2^3": {"design": "factorial", "factors": 3},
    "fractional 2^(5-1)": {"design": "factorial", "factors": 5, "options": {"fraction": 2}},
    "Plackett-Burman": {"design": "pb", "factors": 7},
    "min-run res V": {"design": "mr5", "factors": 6},
    "CCD": {"design": "ccd", "factors": 3},
    "Box-Behnken": {"design": "bbd", "factors": 3},
    "3-level factorial": {"design": "threelevel", "factors": 2},
    "D-optimal": {"design": "dopt", "factors": 3},
    "definitive screening": {"design": "dsd", "factors": 5},
    "one factor": {"design": "onefactor", "factors": 1},
    "Latin hypercube": {"design": "lhs", "factors": 3},
    "general factorial": {"design": "general", "factors": 2, "categoric": 3},
    "Taguchi": {"design": "taguchi", "factors": 4},
    "simplex lattice": {"design": "lattice", "factors": 3, "mixture": True},
    "simplex centroid": {"design": "centroid", "factors": 3, "mixture": True},
    "mixture optimal": {"design": "mixdopt", "factors": 3, "mixture": True},
}


def build(name: str, spec: dict):
    from rsstatchem.ui.dx.document import DesignDocument

    k = spec["factors"]
    factors = []
    for i in range(k):
        f = {"name": NAMES[i]}
        if spec.get("mixture"):
            f.update(type="mixture", low=0.0, high=1.0)
        elif spec.get("categoric"):
            f.update(type="categoric", levels=[f"L{j + 1}" for j in range(spec["categoric"])])
        factors.append(f)
    full = {"design": spec["design"], "randomize": False, "factors": factors,
            "responses": [{"name": "Yield"}, {"name": "Purity"}],
            "options": dict(spec.get("options", {}))}
    doc = DesignDocument.from_spec(full)
    rng = np.random.default_rng(len(name))
    coded = np.nan_to_num(doc.coded_matrix().astype(float))
    for j, resp in enumerate(doc.responses):
        y = 50 + 5 * j + coded @ rng.normal(2, 1, coded.shape[1]) \
            - 1.5 * (coded[:, 0] ** 2) + rng.normal(0, 0.8, len(coded))
        doc.ds.df[resp["column"]] = np.round(y, 2)
    return doc


def visit(app, window, where: str) -> int:
    seen = 0
    for key in list(window.nav_items):
        crawler.current.update(module="rsm", command=f"{where} | node {key}")
        try:
            page = window.show_node(key)
            app.processEvents()
            seen += 1
            strip = getattr(page, "strip", None)
            if strip is not None:
                for index, label in enumerate(strip.labels()):
                    crawler.current["command"] = f"{where} | node {key} | tab {label}"
                    strip.set_current(index)       # what a click on the tab does
                    app.processEvents()
                    seen += 1
                    # every entry of the combo boxes on this tab (graph type, term, ...)
                    for combo in page.findChildren(QComboBox):
                        if not combo.isVisible() or not combo.isEnabled() \
                                or combo.count() > 25:
                            continue
                        start = combo.currentIndex()
                        for i in range(combo.count()):
                            crawler.current["command"] = (
                                f"{where} | node {key} | tab {label} | "
                                f"{combo.objectName() or 'combo'} = {combo.itemText(i)}")
                            combo.setCurrentIndex(i)
                            app.processEvents()
                            seen += 1
                        combo.setCurrentIndex(start)
                        app.processEvents()
        except Exception:  # noqa: BLE001
            crawler.note("exception", traceback.format_exc())
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k")
    args = ap.parse_args()

    from rsstatchem.app import AppContext
    from rsstatchem.ui.dx.examples import EXAMPLES
    from rsstatchem.ui.dx.window import DXMainWindow
    from rsstatchem.ui.theme import apply_theme

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    ctx = AppContext(app)
    ctx.app.quit = lambda: None
    crawler.install_patches(app)

    jobs = [(f"example {key}", fn) for key, (_text, fn) in EXAMPLES.items()]
    jobs += [(f"design {name}", (lambda n=name, s=spec: build(n, s)))
             for name, spec in SPECS.items()]
    for where, make in jobs:
        if args.k and args.k.lower() not in where.lower():
            continue
        crawler.current.update(module="rsm", command=f"{where} | create")
        before = len(crawler.events)
        try:
            doc = make()
            window = DXMainWindow(ctx, doc)
            window.show()
            app.processEvents()
            seen = visit(app, window, where)
            window.doc.modified = False
            window.hide()
        except Exception:  # noqa: BLE001
            crawler.note("exception", traceback.format_exc())
            seen = 0
        bad = [e for e in crawler.events[before:] if e["kind"] != "qt"]
        print(f"{where:32} {seen:4d} screens, {len(bad)} problem(s)", flush=True)
    for e in crawler.events:
        if e["kind"] == "qt":
            continue
        lines = e["text"].strip().splitlines()
        print(f"\n{e['command']} :: {e['kind']}\n  "
              + "\n  ".join(lines[-5:] if e["kind"] == "exception" else lines[:3]))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os

    os._exit(code)
