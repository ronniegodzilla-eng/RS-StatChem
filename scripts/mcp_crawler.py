"""Debugging aid: run every menu command of a module through the Claude connector's engine.

    python scripts/mcp_crawler.py [stats rsm graph chem image output] [--json report.json]
                                  [-k text] [--start N]

For every command: ``run_command``; a dialog it opens is described, analysis dialogs of the
Statistics module are filled with plausible variables and answered with OK, any other dialog is
cancelled. Reported per command: ``output`` (a result came back), ``dialog`` (opened, described
and closed), ``done``, ``refused`` (a message for the user), ``crash`` (an exception), ``stuck``
(a dialog could not be closed, or the time limit passed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["RSSTATCHEM_HOME"] = tempfile.mkdtemp(prefix="rsmcp-")
os.environ["RSSTATCHEM_NO_SPLASH"] = "1"

LIMIT = 90
CLOSERS = ("Cancel", "Close", "No", "Don't Save", "Discard", "OK", "Continue")


def fill_plan(dialog: dict, ds) -> dict:
    """Plausible variables for the boxes of a described analysis dialog."""
    from analysis_fuzzer import DEPENDENT_KEYS, GROUP_KEYS

    plan, used = {}, set()
    for box in dialog["variables"]["boxes"]:
        key = box["box"].lower()
        if box["holds"].startswith("pairs"):
            plan[box["box"]] = [["y", "pre"]]
            continue
        if any(k in key for k in GROUP_KEYS) and key not in ("y", "dv"):
            order = ["sex", "grp", "flag", "likert1", "x1", "x2"]
        elif any(k in key for k in DEPENDENT_KEYS):
            order = ["y", "time", "flag", "x1", "x2", "cnt"]
        else:
            order = ["x1", "x2", "x3", "likert1", "likert2", "likert3", "y", "pre", "cnt", "grp",
                     "sex", "flag", "txt"]
        names = [n for n in order if n not in used and n in dialog["variables"]["available"]]
        plan[box["box"]] = (names[:1] if box["holds"] == "one variable" else names[:3])
        used.update(plan[box["box"]])
    return plan


class Crawler:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.results: list[dict] = []

    def call(self, name, **args):
        parts = self.engine.call(name, args)
        texts = [p for p in parts if isinstance(p, str)]
        try:
            return json.loads(texts[0]), parts
        except (ValueError, IndexError):
            return {}, parts

    def close_dialogs(self) -> bool:
        """Cancel whatever is open; True when nothing is left."""
        for _ in range(8):
            state, _ = self.call("app_status")
            if not state.get("dialog"):
                return True
            d, _ = self.call("read_window")
            labels = [b if isinstance(b, str) else b["label"] for b in d.get("buttons", [])]
            pick = next((c for c in CLOSERS if c in labels), None)
            try:
                if pick:
                    self.call("fill_dialog", press=pick, timeout=20)
                else:
                    self.call("close_window", window=d["window"])
            except Exception:  # noqa: BLE001 - try the close button
                try:
                    self.call("close_window", window=d["window"])
                except Exception:  # noqa: BLE001
                    return False
        return False

    def prepare(self, module: str) -> None:
        from analysis_fuzzer import demo_dataset

        ws = self.engine.workspace

        def setup():
            from menu_crawler import demo

            if module == "stats":
                ds = demo_dataset()
                for e in list(ws.ctx.editors):
                    e.ds.modified = False
                blank = ws.ctx.editors[-1] if ws.ctx.editors else None
                ws._last_window = blank
                if blank is not None:
                    blank.load_dataset(ds)
                    blank.ds.modified = False
                else:
                    ws._last_window = ws.ctx.new_editor(ds)
            elif module == "output":
                ws.ctx.show_output_window()
                ws._last_window = ws.ctx.output()
            else:
                ws._last_window = demo(ws.ctx, module)
        self.engine.gui.call(setup)

    def one(self, module: str, command: str) -> dict:
        r = {"module": module, "command": command, "status": "done", "detail": ""}
        t0 = time.time()
        try:
            d, parts = self.call("run_command", command=command, window=module, timeout=LIMIT)
            if d.get("status", "").startswith("still"):
                d, parts = self.call("wait", timeout=LIMIT)
            if d.get("status", "").startswith("still"):
                return {**r, "status": "stuck", "detail": "still computing"}
            if d.get("errors"):
                return {**r, "status": "crash", "detail": d["errors"][-1]}
            dialog = d.get("dialog")
            if dialog and "variables" in dialog:
                plan = fill_plan(dialog, None)
                d, parts = self.fill(plan)
                if d.get("errors"):
                    return {**r, "status": "crash", "detail": d["errors"][-1]}
                r["status"] = "output" if d.get("output") else "dialog"
                if d.get("dialog"):
                    r["detail"] = str(d["dialog"].get("message") or d["dialog"]["title"])[:200]
                    if "message" in d["dialog"]:
                        r["status"] = "refused"
            elif dialog:
                r["status"] = "refused" if "message" in dialog else "dialog"
                r["detail"] = str(dialog.get("message", dialog.get("title")))[:160]
            elif d.get("output"):
                r["status"] = "output"
        except ValueError as exc:
            r.update(status="refused", detail=str(exc)[:200])
        except Exception as exc:  # noqa: BLE001
            r.update(status="crash", detail=f"{type(exc).__name__}: {exc}"[:400])
        r["seconds"] = round(time.time() - t0, 1)
        try:
            if not self.close_dialogs():
                r.update(status="stuck", detail="a dialog stays open: " + r["detail"])
        except Exception as exc:  # noqa: BLE001
            r.update(status="stuck", detail=f"closing: {exc}"[:300])
        return r

    def fill(self, plan):
        for _ in range(4):
            try:
                return self.call("fill_dialog", variables=plan, press="OK", timeout=LIMIT)
            except ValueError as exc:
                text = str(exc)
                if "does not accept" in text:
                    bad = text.split("accept '")[1].split("'")[0]
                    for k, v in list(plan.items()):
                        if bad in v:
                            alt = [n for n in ("x1", "x2", "y", "cnt", "grp", "sex", "flag", "txt")
                                   if all(n not in u for u in plan.values())]
                            plan[k] = [n if n != bad else (alt[0] if alt else None) for n in v]
                            plan[k] = [n for n in plan[k] if n]
                    continue
                raise
        raise ValueError("no acceptable variables")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modules", nargs="*",
                    default=["stats", "rsm", "graph", "chem", "image", "output"])
    ap.add_argument("--json")
    ap.add_argument("-k", default="")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    from rsstatchem.mcp import server

    app = server._offscreen_app()
    import socket

    def no_network(*a, **k):
        raise OSError("network disabled in the crawler")

    socket.create_connection = no_network
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl = staticmethod(lambda url: True)
    code = {"v": 0}

    def work():
        engine = server._background_engine()
        c = Crawler(engine)
        for module in args.modules:
            c.prepare(module)
            data, _ = c.call("app_status")
            from rsstatchem.mcp import driver

            paths = engine.gui.call(lambda: [p for p, _a in driver.commands(
                engine.workspace._menu_window(module))])
            for i, path in enumerate(paths):
                if i < args.start or args.k.lower() not in path.lower():
                    continue
                last = path.split(" > ")[-1]
                if last.lower() in driver.SKIP_COMMANDS or driver.NATIVE_COMMANDS.match(last):
                    continue
                print(f"#{module} {i} {path}", file=sys.stderr, flush=True)
                try:
                    alive = engine.gui.call(lambda: any(
                        driver.window_kind(w) == module and driver.commands(w)
                        for w in driver.open_windows()))
                    if module == "stats" or not alive:
                        c.prepare(module)
                except Exception as exc:  # noqa: BLE001
                    print("prepare", exc, file=sys.stderr)
                r = c.one(module, path)
                c.results.append(r)
                if r["status"] in ("crash", "stuck"):
                    print(f"{r['status']:7} {module} {path} :: {r['detail'].splitlines()[-1][:200]}",
                          flush=True)
                if args.json:
                    Path(args.json).write_text(json.dumps(c.results, indent=1,
                                                          ensure_ascii=False), encoding="utf-8")
            counts: dict = {}
            for r in c.results:
                if r["module"] == module:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
            print(f"{module}: {sum(counts.values())} commands {counts}", flush=True)
        sys.stdout.flush()
        os._exit(code["v"])     # windows are in arbitrary states: skip Qt teardown

    threading.Thread(target=work, daemon=True).start()
    return app.exec()


if __name__ == "__main__":
    main()
