"""Headless self-test of the Image Analysis module (``RS-StatChem --selftest``).

Checks the Phase 0 contracts and that the bundled readers' dependencies (tifffile, roifile,
pydicom) are importable, then runs ``selftest()`` of every ``commands_<area>`` module that
defines one (work packages add their own checks there; no Qt widgets may be created).
"""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np


def run() -> None:
    import pydicom  # noqa: F401 - bundled for File > Open of DICOM
    import roifile  # noqa: F401 - bundled for .roi / RoiSet.zip
    import tifffile

    from rsstatchem.core.imaging import commands as C
    from rsstatchem.core.imaging.image import GRAY16, ImageDoc
    from rsstatchem.core.imaging.io import open_image
    from rsstatchem.core.imaging.lut import render
    from rsstatchem.core.imaging.menu_layout import AREAS, iter_items
    from rsstatchem.core.imaging.results import ResultsTable
    from rsstatchem.core.imaging.roi import OvalRoi, RectRoi
    from rsstatchem.core.imaging.session import Session

    img = ImageDoc.from_array(np.arange(32 * 32, dtype=np.uint16).reshape(32, 32), "YX", "t")
    assert img.type == GRAY16 and img.display_range == (0.0, 1023.0)
    assert render(img).shape == (32, 32, 3)
    assert RectRoi(2, 3, 4, 5).get_mask((32, 32)).sum() == 20
    assert OvalRoi(0, 0, 10, 10).get_mask((32, 32)).sum() > 60
    opts = C.values_from_options([C.number("Sigma (Radius):", 2.0), C.checkbox("Stack")],
                                 "sigma=3.5 stack")
    assert opts == {"sigma": 3.5, "stack": True}
    assert len({item.label for _, item in iter_items()}) > 350

    session = Session()
    session.registry.add("Selftest Invert", lambda s, im: im.pixels.__ixor__(0xFFFF))
    session.add_image(img)
    session.run("Selftest Invert")
    assert img.raw_value(0, 0) == 65535.0 and img.can_undo()

    rt = ResultsTable.results()
    rt.add_row({"Area": 12, "Mean": 1.23456})
    with tempfile.TemporaryDirectory() as tmp:
        rt.save(Path(tmp) / "r.csv")
        assert ResultsTable.read(Path(tmp) / "r.csv").get_value("Mean", 0) == 1.235
        tifffile.imwrite(Path(tmp) / "a.tif", np.zeros((3, 8, 8), np.uint8), imagej=True,
                         metadata={"axes": "ZYX"})
        assert open_image(Path(tmp) / "a.tif").n_slices == 3
    for area in AREAS.values():
        try:
            module = importlib.import_module(area.module)
        except ModuleNotFoundError as exc:
            if exc.name != area.module:
                raise
            continue
        fn = getattr(module, "selftest", None)
        if fn is not None:
            fn()
