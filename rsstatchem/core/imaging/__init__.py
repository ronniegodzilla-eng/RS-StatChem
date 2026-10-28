"""Image Analysis core (ImageJ 1.54 / Fiji style). No Qt in this package.

Phase 0 contracts (read-only for the work packages; see ``docs/IMAGEJ_CONTRACTS.md``):

* :mod:`.image`       ``ImageDoc``: pixels in TZCYX(S) order, image types, display ranges, LUTs,
                      threshold, ROI, overlay, slice labels, properties, single-level undo
* :mod:`.calibration` ``Calibration``: pixel size and units, origin, value calibration
* :mod:`.lut`         ``LUT`` and the 8-bit/RGB display rendering used by the canvas
* :mod:`.options`     ``Options``: global ImageJ preferences (black background, conversions, ...)
* :mod:`.roi`         ROI data classes (ImageJ types and coordinates) and the mask contract
* :mod:`.results`     ``ResultsTable`` with ImageJ column naming, number formatting and CSV
* :mod:`.commands`    command registry, ``run(label, "key=value ...")`` option strings, dialog
                      field specs (GenericDialog equivalent) and the macro recorder
* :mod:`.session`     ``Session``: open images, active image, Results, ROI Manager, log, events
* :mod:`.menu_layout` the ImageJ 1.54 menu tree with the owner of every command

ImageJ 1.x is public domain (NIH); code ported from it carries an attribution header naming the
Java source file. Fiji plugins are GPL/LGPL and are re-implemented from the papers instead.
"""
from __future__ import annotations

from rsstatchem.core.imaging.calibration import Calibration
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc,
                                           Threshold)
from rsstatchem.core.imaging.lut import LUT
from rsstatchem.core.imaging.options import Options
from rsstatchem.core.imaging.results import ResultsTable

__all__ = ["Calibration", "ImageDoc", "Threshold", "LUT", "Options", "ResultsTable",
           "GRAY8", "GRAY16", "GRAY32", "COLOR_256", "COLOR_RGB"]
