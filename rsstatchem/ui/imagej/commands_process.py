"""Image Analysis commands: Process menu and Image > Adjust > Threshold... / Color Threshold....
**Owned by WP-B.**

``register(registry)`` adds this area's commands (labels exactly as in
``rsstatchem.core.imaging.menu_layout.labels_of_area("process")``) plus the Fiji-style extras
(Auto Threshold, Auto Local Threshold, Enhance Local Contrast (CLAHE), MorphoLibJ-lite), which
are reachable through Find Commands and macros. The command functions live in the per-area
modules ``rsstatchem.ui.imagej.dialogs.process_*`` (headless; Qt is imported lazily by the
window-opening ``ui_action`` callables only). ``selftest()`` (no Qt widgets) is run by
``RS-StatChem --selftest``. See docs/IMAGEJ_CONTRACTS.md.
"""
from __future__ import annotations

import importlib

# per-area modules, each with register(registry) and optionally selftest()
MODULES = (
    "rsstatchem.ui.imagej.dialogs.process_filters",
    "rsstatchem.ui.imagej.dialogs.process_threshold",
    "rsstatchem.ui.imagej.dialogs.process_binary",
    "rsstatchem.ui.imagej.dialogs.process_math",
    "rsstatchem.ui.imagej.dialogs.process_fft",
    "rsstatchem.ui.imagej.dialogs.process_background",
    "rsstatchem.ui.imagej.dialogs.process_batch",
    "rsstatchem.ui.imagej.dialogs.process_fiji",
)


def _modules():
    for name in MODULES:
        try:
            yield importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise


def register(registry) -> None:
    for module in _modules():
        module.register(registry)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    session = Session()
    register(session.registry)
    for module in _modules():
        fn = getattr(module, "selftest", None)
        if fn is not None:
            fn()
