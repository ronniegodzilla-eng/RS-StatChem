"""Tutorial data (Help > Tutorial Data): textbook designs with their measured responses.

* Chemical process (Montgomery, Design and Analysis of Experiments, Example 11.2): central
  composite design in reaction time (80-90 min) and temperature (170-180 °C), alpha 1.414,
  five centre points, response yield (%).
* Filtration rate (Montgomery Example 6.2): 2^4 factorial in temperature, pressure,
  formaldehyde concentration and stirring rate, response filtration rate (gal/h).
"""
from __future__ import annotations

import numpy as np

from rsstatchem.ui.dx.document import DesignDocument

_A = 1.414
# (coded time, coded temperature) -> yield, Montgomery Table 11.6; centre runs in order
_CCD = [((-1, -1), 76.5), ((-1, 1), 77.0), ((1, -1), 78.0), ((1, 1), 79.5), ((0, 0), 79.9),
        ((0, 0), 80.3), ((0, 0), 80.0), ((0, 0), 79.7), ((0, 0), 79.8), ((_A, 0), 78.4),
        ((-_A, 0), 75.6), ((0, _A), 78.5), ((0, -_A), 77.0)]
_FILTRATION = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]


def chemical_process(with_data: bool = True) -> DesignDocument:
    spec = {"design": "ccd", "name": "Chemical process",
            "factors": [{"name": "Time", "units": "min", "low": 80, "high": 90},
                        {"name": "Temperature", "units": "°C", "low": 170, "high": 180}],
            "responses": [{"name": "Yield", "units": "%"}],
            "options": {"alpha_type": "custom", "alpha": _A, "center_points": 5},
            "randomize": True, "seed": 11}
    doc = DesignDocument.from_spec(spec)
    if with_data:
        coded = doc.coded_matrix()
        pool = list(_CCD)
        changes = []
        for row in np.argsort(doc.ds.df["Std"].to_numpy(float)):
            pt = coded[row]
            k = next(i for i, (c, _) in enumerate(pool) if np.allclose(c, pt, atol=1e-6))
            changes.append((int(row), doc.response_cols[0], pool.pop(k)[1]))
        doc.set_values(changes)
    doc.notes = ("Montgomery, Design and Analysis of Experiments, Example 11.2: maximise the "
                 "yield of a chemical process. Expected result: quadratic model, stationary "
                 "point at Time ≈ 86.9 min, Temperature ≈ 176.5 °C, yield ≈ 80.2 %.")
    doc.modified = False
    return doc


def filtration_rate(with_data: bool = True) -> DesignDocument:
    spec = {"design": "factorial", "name": "Filtration rate",
            "factors": [{"name": "Temperature", "units": "°C", "low": 24, "high": 35},
                        {"name": "Pressure", "units": "psig", "low": 10, "high": 15},
                        {"name": "Concentration", "units": "%", "low": 2, "high": 4},
                        {"name": "Stirring", "units": "rpm", "low": 15, "high": 30}],
            "responses": [{"name": "Filtration rate", "units": "gal/h"}],
            "options": {}, "randomize": True, "seed": 7}
    doc = DesignDocument.from_spec(spec)
    if with_data:
        std = doc.ds.df["Std"].to_numpy(int)
        doc.set_values([(i, doc.response_cols[0], float(_FILTRATION[s - 1]))
                        for i, s in enumerate(std)])
    doc.notes = ("Montgomery, Design and Analysis of Experiments, Example 6.2: pilot-plant "
                 "filtration rate. Significant effects: A, C, D, AC and AD.")
    doc.modified = False
    return doc


EXAMPLES = {"chemical": ("Chemical process (CCD, Montgomery 11.2)", chemical_process),
            "filtration": ("Filtration rate (2⁴ factorial, Montgomery 6.2)", filtration_rate)}
