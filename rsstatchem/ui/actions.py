"""Connecting menu actions without Qt's ``checked`` flag leaking into slot arguments.

``QAction.triggered`` carries a bool. PySide passes it to any slot that can take a positional
argument, so a method such as ``save_as(self, path=None)`` connected directly is called as
``save_as(False)``: File > Save As then "saves" to the path ``False``. :func:`connect_triggered`
is what the action helpers of the module windows use instead of ``triggered.connect(slot)``.
"""
from __future__ import annotations

import inspect
from typing import Callable

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def flag_use(slot: Callable) -> str:
    """How ``slot`` relates to the checked flag: ``"required"`` (its first parameter has no
    default, so it was written to receive the flag), ``"optional"`` (it has a default: the
    flag is passed only for checkable actions) or ``"none"`` (no positional parameter)."""
    try:
        params = list(inspect.signature(slot).parameters.values())
    except (TypeError, ValueError):
        return "none"
    for p in params:
        if p.kind in _POSITIONAL:
            return "required" if p.default is inspect.Parameter.empty else "optional"
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return "optional"
    return "none"


def connect_triggered(action, slot: Callable) -> None:
    """Call ``slot`` when ``action`` is triggered; the checked state is passed only to a slot
    that needs it (a required first parameter) or, for a checkable action, can take it."""
    use = flag_use(slot)

    def run(checked: bool = False):
        if use == "required" or (use == "optional" and action.isCheckable()):
            return slot(checked)
        return slot()

    action.triggered.connect(run)
