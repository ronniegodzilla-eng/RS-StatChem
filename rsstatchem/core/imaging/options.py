"""Global ImageJ preferences (``ij.Prefs`` + the Edit > Options / Set Measurements state).

One :class:`Options` instance lives on the :class:`~rsstatchem.core.imaging.session.Session`.
Core functions take the values they need as arguments (so they stay testable); commands read
them from ``session.options``. The UI persists the whole object with :meth:`to_dict` /
:meth:`from_dict` under the QSettings group "imagej". Defaults are ImageJ 1.54's (not Fiji's:
``black_background`` is False in ImageJ, True after a Fiji install - golden cases set it
explicitly). Add new fields with defaults only; unknown keys in stored settings are ignored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

# Analyze > Set Measurements keys (macro keys of the dialog, in dialog order)
MEASUREMENTS = ("area", "mean", "standard", "modal", "min", "centroid", "center",
                "perimeter", "bounding", "fit", "shape", "feret's", "integrated", "median",
                "skewness", "kurtosis", "area_fraction", "stack", "limit", "display", "invert",
                "scientific", "add", "nan")


@dataclass
class Options:
    # Process > Binary > Options...
    black_background: bool = False
    binary_iterations: int = 1
    binary_count: int = 1
    pad_edges: bool = False
    edm_output: str = "Overwrite"          # Overwrite | 8-bit | 16-bit | 32-bit
    # Edit > Options > Conversions...
    scale_conversions: bool = True
    weighted_rgb: bool = False
    full_range_16bit_inversion: bool = False
    # Edit > Options > Misc...
    divide_by_zero: float = float("inf")
    require_control_key: bool = False
    move_isolated: bool = False
    # Edit > Options > Input/Output...
    jpeg_quality: int = 85
    results_extension: str = ".csv"
    save_column_headers: bool = True
    save_row_numbers: bool = True
    intel_byte_order: bool = False
    # Edit > Options > Colors... (hex "#rrggbb")
    foreground: str = "#ffffff"
    background: str = "#000000"
    selection: str = "#ffff00"
    # Edit > Options > Line Width... / Wand / Point tool / Roi defaults
    line_width: float = 1.0
    wand_mode: str = "Legacy"              # Legacy | 4-connected | 8-connected
    wand_tolerance: float = 0.0
    wand_smooth: bool = False
    point_type: str = "Hybrid"
    point_color: str = "Yellow"
    point_size: str = "Small"
    point_auto_measure: bool = False
    point_add_to_manager: bool = False
    point_label: bool = False
    rounded_rect_diameter: int = 0
    # Analyze > Set Measurements...
    measurements: tuple = ("area", "mean", "min")
    redirect: str = "None"
    decimal_places: int = 3
    # Image > Adjust > Threshold / auto threshold defaults
    threshold_method: str = "Default"
    # Display / misc
    interpolate_zoomed: bool = False
    extra: dict = field(default_factory=dict)   # free-form values for WP-specific options

    def measure(self, key: str) -> bool:
        """Is a Set Measurements flag on? (keys from :data:`MEASUREMENTS`)."""
        return key in self.measurements

    def to_dict(self) -> dict:
        d = asdict(self)
        d["measurements"] = list(self.measurements)
        return d

    @classmethod
    def from_dict(cls, data: dict | None) -> "Options":
        opts = cls()
        if not data:
            return opts
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            default = getattr(opts, f.name)
            try:
                if isinstance(default, bool):
                    value = value in (True, "true", "True", 1, "1")
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
                elif isinstance(default, tuple):
                    value = tuple([value] if isinstance(value, str) else value)
                elif isinstance(default, dict):
                    value = dict(value)
            except (TypeError, ValueError):
                continue
            setattr(opts, f.name, value)
        return opts
