"""Helpers for tests that compare with golden outputs of the real ImageJ 1.54p
(``tests/data_imagej``, produced by ``scripts/imagej_golden.py``; no Java needed here).

    from ij_golden import golden_csv, golden_images, input_path, thresholds

``golden_csv("particles", "blobs_default.csv")`` -> ResultsTable, ``golden_images("filters")``
-> dict of arrays (npz), ``thresholds()`` -> {(image, method, dark): (lower, upper)}.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data_imagej"
INPUTS = DATA / "inputs"
GOLDEN = DATA / "golden"


def input_path(name: str) -> Path:
    return INPUTS / name


def golden_path(case: str, name: str) -> Path:
    return GOLDEN / case / name


def golden_csv(case: str, name: str):
    from rsstatchem.core.imaging.results import ResultsTable
    return ResultsTable.read(golden_path(case, name))


def golden_images(case: str) -> dict[str, np.ndarray]:
    with np.load(golden_path(case, "images.npz")) as z:
        return {k: z[k] for k in z.files}


def thresholds() -> dict[tuple[str, str, bool], tuple[float, float]]:
    out = {}
    with open(golden_path("thresholds", "thresholds.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["image"], row["method"], row["dark"] == "1")] = (float(row["lower"]),
                                                                    float(row["upper"]))
    return out


def blobs():
    """blobs.gif as ImageJ opens it (8-bit, inverting LUT)."""
    from rsstatchem.core.imaging.io import open_image
    return open_image(input_path("blobs.gif"))
