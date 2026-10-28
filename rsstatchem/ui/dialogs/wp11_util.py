"""Small helpers shared by the WP11 dialogs (Data Preparation, Quality Control, Direct
Marketing, Conjoint and the SPSS 31/32 extensions)."""
from __future__ import annotations

from rsstatchem.core.dataset import NUMERIC, Dataset


def open_dataset(dialog, ds: Dataset) -> bool:
    """Show ``ds`` in a new Data Editor window when the dialog was opened from an editor;
    otherwise keep it on the dialog (``dialog.new_dataset``) for the caller / tests."""
    dialog.new_dataset = ds
    editor = dialog.parent()
    ctx = getattr(editor, "ctx", None)
    if ctx is not None and hasattr(ctx, "new_editor"):
        ctx.new_editor(ds)
        return True
    return False


def parse_values(text: str, ds: Dataset | None = None, var: str | None = None) -> list:
    """Values typed by the user ("1 2", "1;2", "a, b"): numbers for numeric variables."""
    raw = [t for t in str(text or "").replace(";", " ").replace(",", " ").split() if t]
    numeric = var is None or ds is None or ds.var(var).type == NUMERIC
    out = []
    for t in raw:
        if numeric:
            try:
                out.append(float(t))
                continue
            except ValueError:
                pass
        out.append(t)
    return out


def number_or_none(text) -> float | None:
    t = str(text or "").strip().replace(",", ".")
    if not t:
        return None
    return float(t)


def editor_datasets(dialog) -> list[Dataset]:
    """Datasets of the other open Data Editor windows (for plan files and the like)."""
    editor = dialog.parent()
    ctx = getattr(editor, "ctx", None)
    out = []
    for ed in getattr(ctx, "editors", []) or []:
        ds = getattr(ed, "ds", None)
        if ds is not None and ds is not getattr(dialog, "ds", None):
            out.append(ds)
    return out
