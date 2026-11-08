"""Record where the app is stuck when its interface stops responding.

A Qt timer marks the main thread as alive once a second (nested event loops such as open
dialogs keep it running). A background thread checks the mark; when the main thread has been
silent for ``limit`` seconds it writes the Python stack of every thread to
``<user data>/logs/hang-<time>.txt`` once per episode, so an intermittent freeze can be traced
from the file instead of guessed at. Nothing is sent anywhere.
"""
from __future__ import annotations

import datetime as _dt
import faulthandler
import threading
import time
from pathlib import Path


class HangWatchdog:
    def __init__(self, log_dir: Path, limit: float = 10.0, interval_ms: int = 1000):
        from PySide6.QtCore import QTimer

        self.log_dir = Path(log_dir)
        self.limit = limit
        self.last_beat = time.monotonic()
        self.reported = False
        self.files: list[Path] = []
        self._stop = threading.Event()
        self._timer = QTimer()
        self._timer.timeout.connect(self.beat)
        self._timer.start(interval_ms)
        self._thread = threading.Thread(target=self._watch, name="hang-watchdog", daemon=True)
        self._thread.start()

    def beat(self) -> None:
        self.last_beat = time.monotonic()
        self.reported = False

    def _watch(self) -> None:
        while not self._stop.wait(min(1.0, self.limit / 4)):
            silent = time.monotonic() - self.last_beat
            if silent >= self.limit and not self.reported:
                self.reported = True
                self.write_report(silent)

    def write_report(self, silent: float) -> Path | None:
        try:
            from rsstatchem import __version__

            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.log_dir / f"hang-{stamp}-{len(self.files) + 1}.txt"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"RS-StatChem {__version__}: the interface did not respond for "
                         f"{silent:.0f} s. Stacks of all threads (most recent call last):\n\n")
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
            self.files.append(path)
            return path
        except OSError:
            return None

    def stop(self) -> None:
        self._stop.set()
        self._timer.stop()
