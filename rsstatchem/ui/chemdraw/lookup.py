"""Background PubChem lookups (Qt thread pool, like the update checker) so the UI never waits
on the network."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThreadPool, Signal

from rsstatchem.core.chem_draw import names

__all__ = ["NameLookup"]


class NameLookup(QObject):
    """Runs ``names.pubchem_name`` / ``names.pubchem_structure`` off the UI thread and emits
    ``finished(kind, query, NameResult | None, error message | None)`` on the UI thread."""

    finished = Signal(str, str, object, object)

    def __init__(self, parent=None, fetch=None):
        super().__init__(parent)
        self.fetch = fetch          # tests inject a fake network
        self.busy = 0

    def start(self, kind: str, query: str) -> None:
        self.busy += 1
        QThreadPool.globalInstance().start(lambda: self._work(kind, query))

    def _work(self, kind: str, query: str) -> None:  # thread pool
        kw = {"fetch": self.fetch} if self.fetch is not None else {}
        try:
            fn = names.pubchem_name if kind == "name" else names.pubchem_structure
            result, error = fn(query, **kw), None
        except names.NameLookupError as exc:
            result, error = None, str(exc)
        except Exception as exc:  # noqa: BLE001 - never crash a pool thread
            result, error = None, f"Unexpected error: {exc}"
        try:
            self.finished.emit(kind, query, result, error)
        except RuntimeError:        # the window was closed while the request was running
            pass
