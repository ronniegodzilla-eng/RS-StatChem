"""ROIs: data classes (:mod:`.shapes`, Phase 0 contract) and their rasterisation (:mod:`.mask`,
owned by WP-C). WP-C adds ``geometry.py``, ``wand.py``, ``roi_io.py`` and ``selection_ops.py``."""
from __future__ import annotations

from rsstatchem.core.imaging.roi.shapes import (ANGLE, COMPOSITE, FREELINE, FREEROI, LINE, OVAL,
                                                POINT, POLYGON, POLYLINE, RECTANGLE, TRACED_ROI,
                                                CompositeRoi, EllipseRoi, LineRoi, OvalRoi,
                                                Overlay, PointRoi, PolygonRoi, RectRoi, Roi,
                                                RotatedRectRoi, TextRoi)

__all__ = ["Roi", "RectRoi", "OvalRoi", "PolygonRoi", "LineRoi", "PointRoi", "RotatedRectRoi",
           "EllipseRoi", "TextRoi", "CompositeRoi", "Overlay", "RECTANGLE", "OVAL", "POLYGON",
           "FREEROI", "TRACED_ROI", "LINE", "POLYLINE", "FREELINE", "ANGLE", "COMPOSITE", "POINT"]
