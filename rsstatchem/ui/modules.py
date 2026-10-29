"""The five modules of RS-StatChem and how their main windows are opened.

Each module is a separate workspace with its own window style and data entry:

* ``stats``  - SPSS-style Data Editor, Analyze menu and Output Viewer
* ``rsm``    - Design-Expert-style design of experiments / response surfaces
* ``graph``  - OriginLab-style workbooks, graphs and curve fitting
* ``chem``   - ChemDraw-style structure and reaction drawing
* ``image``  - ImageJ-style image processing and measurement

A module package exposes ``open_workspace(ctx, path=None)`` returning its (shown) main
window. Contract for module windows: keep ``self.ctx = ctx``; in ``closeEvent``, once the close
is accepted, call ``ctx.window_closed(self)`` (the app quits when the last window closes); put
:func:`add_module_menu` in the Window menu so users can open the launcher or another module.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleInfo:
    key: str
    title: str
    subtitle: str
    style: str           # the desktop application whose workflow the module follows
    opener: str          # "package.module:function"
    extensions: tuple    # files this module opens (lower case, with dot)
    icon: str            # key in rsstatchem.ui.launcher.TILE_ICONS


MODULES: tuple[ModuleInfo, ...] = (
    ModuleInfo("stats", "Statistics", "Data Editor, descriptive to multivariate analyses, "
               "SEM, output to Word", "SPSS", "rsstatchem.ui.data_editor:open_workspace",
               (".rsd", ".sav", ".por", ".dta", ".sas7bdat", ".xpt", ".csv", ".txt", ".dat",
                ".tsv", ".xlsx", ".xls", ".rso"), "stats"),
    ModuleInfo("rsm", "RSM / DOE", "Factorial, response surface and mixture designs, "
               "ANOVA, model graphs, optimization", "Design-Expert",
               "rsstatchem.ui.dx:open_workspace", (".rsx",), "rsm"),
    ModuleInfo("graph", "Graph & Analysis", "Workbooks, publication graphs, curve fitting, "
               "peak and signal analysis", "OriginLab", "rsstatchem.ui.origin:open_workspace",
               (".rsop", ".rsg"), "graph"),
    ModuleInfo("chem", "Chemical Drawing", "Structures, reactions, names to structures, "
               "chemical properties", "ChemDraw", "rsstatchem.ui.chemdraw:open_workspace",
               (".rsc", ".mol", ".sdf", ".rxn", ".cdxml"), "chem"),
    ModuleInfo("image", "Image Analysis", "Open, process and measure images and stacks: "
               "filters, thresholds, particles, ROIs", "ImageJ",
               "rsstatchem.ui.imagej:open_workspace",
               (".rsij", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
                ".fits", ".roi"), "image"),
)
BY_KEY = {m.key: m for m in MODULES}


def module_for_path(path: str) -> ModuleInfo | None:
    suffix = Path(path).suffix.lower()
    return next((m for m in MODULES if suffix in m.extensions), None)


def open_module(ctx, key: str, path: str | None = None):
    """Open a new main window of module ``key`` (optionally with a file) and return it."""
    info = BY_KEY[key]
    module_name, func = info.opener.split(":")
    opener = getattr(importlib.import_module(module_name), func)
    window = opener(ctx, path=path)
    if window is not None:
        ctx.register_window(window)
        ctx.settings.setValue("last_module", key)
    return window


def open_path(ctx, path: str):
    """Open a file in the module that handles its type (Statistics for anything unknown)."""
    info = module_for_path(path) or BY_KEY["stats"]
    return open_module(ctx, info.key, path=path)


def add_module_menu(menu, ctx):
    """A "Modules" submenu (launcher + one entry per module) appended to ``menu``."""
    sub = menu.addMenu("&Modules")
    sub.addAction("Module &Launcher...", ctx.show_launcher)
    sub.addSeparator()
    for info in MODULES:
        # "&&": a single "&" would be read as a keyboard mnemonic and disappear
        sub.addAction(f"New {info.title.replace('&', '&&')} Window",
                      lambda _=False, k=info.key: open_module(ctx, k))
    return sub
