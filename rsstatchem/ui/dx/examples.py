"""Tutorial data (Help > Tutorial Data): textbook designs with their measured responses.

* Chemical process (Montgomery, Design and Analysis of Experiments, Example 11.2): central
  composite design in reaction time (80-90 min) and temperature (170-180 °C), alpha 1.414,
  five centre points, response yield (%).
* Filtration rate (Montgomery Example 6.2): 2^4 factorial in temperature, pressure,
  formaldehyde concentration and stirring rate, response filtration rate (gal/h).
* Battery life (Montgomery Examples 5.1 / 5.5): temperature (numeric, 3 levels) x plate
  material (categoric, 3 levels), four replicates.
* Paper pulp (Montgomery Section 14.4): split-plot, pulp preparation method hard to change,
  temperature easy to change, three replicates (days) as blocks.
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


# Montgomery Table 14.16 (paper tensile strength): rows = temperature 200..275 °F, columns =
# (replicate 1: pulp 1, 2, 3), (replicate 2: ...), (replicate 3: ...)
_PULP = [[30, 34, 29, 28, 31, 31, 31, 35, 32], [35, 41, 26, 32, 36, 30, 37, 40, 34],
         [37, 38, 33, 40, 42, 32, 41, 39, 39], [36, 42, 36, 41, 40, 40, 40, 44, 45]]
# Montgomery Table 5.1 (battery life, h): material 1..3 x temperature 15, 70, 125 °F
_BATTERY = {(1, 15): [130, 155, 74, 180], (1, 70): [34, 40, 80, 75], (1, 125): [20, 70, 82, 58],
            (2, 15): [150, 188, 159, 126], (2, 70): [136, 122, 106, 115],
            (2, 125): [25, 70, 58, 45], (3, 15): [138, 110, 168, 160],
            (3, 70): [174, 120, 150, 139], (3, 125): [96, 104, 82, 60]}


def paper_pulp(with_data: bool = True) -> DesignDocument:
    """Split-plot: pulp preparation method (hard to change, whole plots = batches) x cooking
    temperature, three replicates (days) as blocks."""
    spec = {"design": "general", "name": "Paper pulp",
            "factors": [{"name": "Pulp Prep", "levels": ["a1", "a2", "a3"], "changes": "hard"},
                        {"name": "Temp", "units": "deg F", "levels": ["200", "225", "250",
                                                                      "275"]}],
            "responses": [{"name": "Tensile", "units": "strength"}],
            "options": {"replicates": 3, "block_per_replicate": True},
            "randomize": True, "seed": 5}
    doc = DesignDocument.from_spec(spec)
    if with_data:
        df = doc.ds.df
        cols = doc.factor_cols
        temps = [200.0, 225.0, 250.0, 275.0]
        changes = []
        for i in range(doc.n_runs):
            rep = int(df.at[i, "Block"]) - 1
            meth = int(round(df.at[i, cols[0]])) - 1
            ti = temps.index(float(df.at[i, cols[1]]))
            changes.append((i, doc.response_cols[0], float(_PULP[ti][rep * 3 + meth])))
        doc.set_values(changes)
    doc.notes = ("Montgomery, Design and Analysis of Experiments, Section 14.4 (paper tensile "
                 "strength): a split-plot with the pulp preparation method hard to change. "
                 "REML / Kenward-Roger reproduces the split-plot ANOVA: pulp F = 7.08 on 2 and "
                 "4 df; temperature and the interaction tested on 18 subplot df.")
    doc.modified = False
    return doc


def battery_life(with_data: bool = True) -> DesignDocument:
    """Numeric temperature (3 levels) crossed with a categoric material, 4 replicates."""
    spec = {"design": "threelevel", "name": "Battery life",
            "factors": [{"name": "Temperature", "units": "deg F", "low": 15, "high": 125},
                        {"name": "Material", "type": "categoric",
                         "levels": ["Type 1", "Type 2", "Type 3"]}],
            "responses": [{"name": "Life", "units": "h"}],
            "options": {"replicates": 4}, "randomize": True, "seed": 9}
    doc = DesignDocument.from_spec(spec)
    if with_data:
        df = doc.ds.df
        t, m = doc.factor_cols
        pools = {k: list(v) for k, v in _BATTERY.items()}
        changes = []
        for i in np.argsort(df["Std"].to_numpy(float)):
            key = (int(df.at[i, m]), int(round(df.at[i, t])))
            changes.append((int(i), doc.response_cols[0], float(pools[key].pop(0))))
        doc.set_values(changes)
    doc.notes = ("Montgomery, Design and Analysis of Experiments, Examples 5.1 and 5.5: battery "
                 "life for three plate materials (categoric) at three temperatures (numeric). "
                 "Error SS 18230.75 on 27 df.")
    doc.modified = False
    return doc


EXAMPLES = {"chemical": ("Chemical process (CCD, Montgomery 11.2)", chemical_process),
            "filtration": ("Filtration rate (2⁴ factorial, Montgomery 6.2)", filtration_rate),
            "battery": ("Battery life (numeric x categoric, Montgomery 5.1)", battery_life),
            "pulp": ("Paper pulp (split-plot, Montgomery 14.4)", paper_pulp)}
