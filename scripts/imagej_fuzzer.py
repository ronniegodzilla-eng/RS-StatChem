"""Debugging aid: run every Image Analysis command with its default options on several kinds
of image and report the ones that raise anything but a CommandError.

    QT_QPA_PLATFORM=offscreen python scripts/imagej_fuzzer.py [--json report.json] [-k text]

CommandError is how a command tells the user it cannot run here ("Stack required", "Area
selection required", ...). Anything else (TypeError, IndexError, ...) is a bug. Each command
runs on: an 8-bit image, the same with a rectangle selection, a binary mask, an RGB image, a
16-bit stack and a 32-bit image. Images are created fresh for every run.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import menu_crawler as crawler  # noqa: E402  (sets the offscreen platform and a temp home)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

LIMIT = 60
SKIP = {"Quit"}


def contexts():
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.io import open_image
    from rsstatchem.core.imaging.roi import RectRoi

    blobs = crawler.ROOT / "tests" / "data_imagej" / "inputs" / "blobs.gif"
    rng = np.random.default_rng(5)

    def gray8():
        return open_image(blobs)

    def with_roi():
        im = open_image(blobs)
        im.roi = RectRoi(40, 30, 90, 70)
        return im

    def mask():
        im = open_image(blobs)
        px = im.plane()
        return ImageDoc.from_array(np.where(px > 125, 255, 0).astype(np.uint8), "YX", "mask")

    def rgb():
        return ImageDoc.from_array(rng.integers(0, 255, (60, 80, 3)).astype(np.uint8), "YXS",
                                   "rgb")

    def stack16():
        return ImageDoc.from_array(rng.integers(0, 4000, (5, 48, 64)).astype(np.uint16), "ZYX",
                                   "stack16")

    def float32():
        return ImageDoc.from_array(rng.normal(10, 3, (48, 64)).astype(np.float32), "YX", "f32")

    return {"8-bit": gray8, "8-bit+ROI": with_roi, "mask": mask, "RGB": rgb,
            "16-bit stack": stack16, "32-bit": float32}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json")
    ap.add_argument("-k")
    args = ap.parse_args()

    from rsstatchem.app import AppContext
    from rsstatchem.core.imaging.commands import CommandError
    from rsstatchem.ui import modules
    from rsstatchem.ui.theme import apply_theme

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    ctx = AppContext(app)
    ctx.app.quit = lambda: None
    ctx.check_for_updates = lambda manual=False: None
    crawler.install_patches(app)
    bar = modules.open_module(ctx, "image")
    session = bar.session
    labels = [lb for lb in session.registry.labels()
              if session.registry.get(lb).func is not None and lb.strip(". ") not in SKIP
              and (not args.k or args.k.lower() in lb.lower())]
    results = []
    makers = contexts()
    for label in labels:
        for name, make in makers.items():
            crawler.current.update(module="image", command=f"{label} [{name}]")
            for im in list(session.images):
                session.close_image(im) if hasattr(session, "close_image") else \
                    session.images.remove(im)
            app.processEvents()
            session.add_image(make())
            before = len(crawler.events)
            status, detail = "ok", ""
            faulthandler.dump_traceback_later(LIMIT, exit=True)
            t0 = time.perf_counter()
            try:
                session.run(label, "")
                app.processEvents()
            except CommandError as exc:
                status, detail = "refused", str(exc)
            except Exception:  # noqa: BLE001
                status, detail = "crash", traceback.format_exc()
            faulthandler.cancel_dump_traceback_later()
            for e in crawler.events[before:]:
                if e["kind"] == "exception":
                    status, detail = "crash", e["text"]
            seconds = round(time.perf_counter() - t0, 2)
            results.append({"command": label, "context": name, "status": status,
                            "detail": detail, "seconds": seconds})
            if status == "crash":
                last = detail.strip().splitlines()[-1]
                print(f"crash  {label!r} on {name}: {last[:170]}", flush=True)
            elif seconds > 15:
                print(f"slow   {label!r} on {name}: {seconds} s", flush=True)
            for w in app.topLevelWidgets():
                if w is not bar and w.isVisible() and w.windowTitle() not in ("",):
                    try:
                        w.close()
                    except RuntimeError:
                        pass
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{len(labels)} commands x {len(makers)} images: {counts}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os

    os._exit(code)
