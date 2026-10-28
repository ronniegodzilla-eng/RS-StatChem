"""Process > Subtract Background... **WP-B.** Headless (no Qt); the algorithm is the exact port
in :mod:`rsstatchem.core.imaging.background`.

Macro keys as in ImageJ: ``rolling=50 light separate create sliding disable stack`` (ImageJ
also accepts the old key "white" for "light"; macro defaults are radius 50 and every checkbox
off). Like ImageJ the whole plane is processed (the filter does not support masking) and an
inverting LUT flips the meaning of "light background".
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging import background as B
from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.ui.imagej.dialogs.process_common import ALL_TYPES


def _fields(session, image):
    fields = [C.number("Rolling ball radius:", 50.0, 1, "pixels"),
              C.checkbox("Light background", False)]
    if image is not None and image.is_rgb:
        fields.append(C.checkbox("Separate colors", False))
    fields += [C.checkbox("Create background (don't subtract)", False),
               C.checkbox("Sliding paraboloid", False),
               C.checkbox("Disable smoothing", False)]
    return fields


def subtract_background(session, image, rolling=50.0, light=False, separate=False,
                        create=False, sliding=False, disable=False, white=False, stack=False):
    radius = float(rolling)
    if radius <= 0.0001:
        raise CommandError("Rolling ball radius must be > 0")
    light = bool(light or white)
    planes = list(image.iter_planes(bool(stack)))
    inverted = image.is_inverted_lut()
    for i, (c, z, t) in enumerate(planes):
        out = B.subtract_background(image.plane(c, z, t), radius, light, bool(separate),
                                    bool(create), bool(sliding), not disable, inverted)
        image.set_plane(out, c, z, t)
        if session is not None and len(planes) > 1:
            session.progress((i + 1) / len(planes))
    image.reset_display_range()


def register(registry) -> None:
    registry.add("Subtract Background...", subtract_background, _fields, supports_stack=True,
                 types=ALL_TYPES)


def selftest() -> None:
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    yy, xx = np.mgrid[0:40, 0:40]
    data = (20 + xx + yy + 100 * (((xx - 20) ** 2 + (yy - 20) ** 2) < 16)).astype(np.uint8)
    img = ImageDoc.from_array(data, "YX", "bg")
    s.add_image(img)
    s.run("Subtract Background...", "rolling=5")
    p = img.plane()
    assert p[20, 20] > 80 and p[2, 2] < 10
    s.run("Subtract Background...", "rolling=5 sliding")
