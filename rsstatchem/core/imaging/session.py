"""The ImageJ session: open images, active image, Results and other tables, ROI Manager, log,
recorder, options, the command registry and change events. No Qt.

The UI (``rsstatchem.ui.imagej.windows``) subscribes to events with :meth:`Session.on` and shows
or updates windows; core code and macros only talk to the session. Events and their argument:

=================  ==========================================================================
"image_added"      ImageDoc newly opened/created (UI opens an image window)
"image_closed"     ImageDoc removed (UI closes its window)
"image_updated"    ImageDoc whose pixels/display/ROI/overlay changed (UI repaints)
"active_changed"   ImageDoc that became active (or None)
"table_updated"    ResultsTable added or changed (UI shows/refreshes its table window)
"log"              str appended to the Log window
"status"           str for the ImageJ bar's status line
"progress"         float 0..1 (1 = done) for the bar's progress indicator
=================  ==========================================================================
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from rsstatchem.core.imaging.commands import CommandRegistry, Recorder
from rsstatchem.core.imaging.image import ImageDoc
from rsstatchem.core.imaging.options import Options
from rsstatchem.core.imaging.results import ResultsTable


class Session:
    def __init__(self, options: Options | None = None) -> None:
        self.options = options or Options()
        self.registry = CommandRegistry()
        self.images: list[ImageDoc] = []
        self.active: ImageDoc | None = None
        self.results = ResultsTable.results("Results")
        self.tables: dict[str, ResultsTable] = {"Results": self.results}
        self.roi_manager: list = []          # list[Roi]; the ROI Manager window shows it
        self.log_lines: list[str] = []
        self.recorder: Recorder | None = None
        self.last_command: tuple[str, str] | None = None
        self.clipboard = None                # ImageDoc (internal clipboard) or None
        self.cancel_requested = False
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------ events
    def on(self, event: str, callback: Callable) -> None:
        self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        if callback in self._listeners.get(event, []):
            self._listeners[event].remove(callback)

    def emit(self, event: str, arg=None) -> None:
        for cb in list(self._listeners.get(event, [])):
            cb(arg)

    # ------------------------------------------------------------------ images
    def add_image(self, image: ImageDoc, activate: bool = True) -> ImageDoc:
        if image not in self.images:
            image.title = self.unique_title(image.title, exclude=image)
            self.images.append(image)
            self.emit("image_added", image)
        if activate:
            self.set_active(image)
        return image

    def close_image(self, image: ImageDoc) -> None:
        if image in self.images:
            self.images.remove(image)
            self.emit("image_closed", image)
        if self.active is image:
            self.set_active(self.images[-1] if self.images else None)

    def set_active(self, image: ImageDoc | None) -> None:
        if image is not self.active:
            self.active = image
            self.emit("active_changed", image)

    def image_updated(self, image: ImageDoc) -> None:
        self.emit("image_updated", image)

    def find_image(self, title_or_id) -> ImageDoc | None:
        """``selectImage`` / ``selectWindow``: by title, by ImageJ-style negative id or by
        1-based index."""
        for im in self.images:
            if im.title == title_or_id or im.id == title_or_id:
                return im
        if isinstance(title_or_id, int) and 1 <= title_or_id <= len(self.images):
            return self.images[title_or_id - 1]
        return None

    def unique_title(self, title: str, exclude: ImageDoc | None = None) -> str:
        """ImageJ ``WindowManager.getUniqueName``: "blobs.gif" -> "blobs-1.gif"."""
        taken = {im.title for im in self.images if im is not exclude}
        if title not in taken:
            return title
        stem, dot, ext = title.rpartition(".")
        if not dot or len(ext) > 5 or " " in ext:
            stem, ext = title, ""
        n = 1
        while True:
            cand = f"{stem}-{n}" + (f".{ext}" if ext else "")
            if cand not in taken:
                return cand
            n += 1

    # ------------------------------------------------------------------ tables & log
    def table(self, title: str = "Results", create: bool = True) -> ResultsTable | None:
        if title not in self.tables and create:
            self.tables[title] = ResultsTable(title)
        return self.tables.get(title)

    def show_table(self, table: ResultsTable) -> None:
        if table.title == "Results":
            self.results = table
        self.tables[table.title] = table
        self.emit("table_updated", table)

    def log(self, text: str) -> None:
        self.log_lines.append(str(text))
        self.emit("log", str(text))

    def status(self, text: str) -> None:
        self.emit("status", text)

    def progress(self, fraction: float) -> None:
        """Commands report progress here; raises Cancelled when the user pressed Esc."""
        self.emit("progress", fraction)
        if self.cancel_requested:
            from rsstatchem.core.imaging.commands import Cancelled
            self.cancel_requested = False
            raise Cancelled()

    # ------------------------------------------------------------------ commands
    def run(self, label: str, options: str | dict | None = "", image: ImageDoc | None = None):
        """Macro-style ``run("Gaussian Blur...", "sigma=2")`` on the active image."""
        return self.registry.run(self, label, options, image=image)

    def handle_result(self, result) -> None:
        if result is None:
            return
        if isinstance(result, (list, tuple)):
            for r in result:
                self.handle_result(r)
            return
        if isinstance(result, ImageDoc):
            self.add_image(result)
        elif isinstance(result, ResultsTable):
            self.show_table(result)

    def record(self, label: str, options: str = "") -> None:
        if self.recorder is not None:
            self.recorder.record(label, options)
