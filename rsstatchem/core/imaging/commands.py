"""Command registry, ImageJ macro option strings, dialog field specs and the recorder.

Every menu command is a :class:`Command` registered under its **exact ImageJ command name**
(the menu label without a letter shortcut, e.g. ``"Gaussian Blur..."``, ``"Properties... "``
with its trailing space, ``"In [+]"``; see :mod:`.menu_layout`). A command declares its dialog
as a list of :class:`Field` (the GenericDialog equivalent). The same fields drive

* the Qt dialog (``rsstatchem.ui.imagej.dialogs.generic.GenericDialog``),
* macro option strings: ``registry.run(session, "Gaussian Blur...", "sigma=2 stack")``,
* the recorder: ``run("Gaussian Blur...", "sigma=2 stack");``.

Option strings follow ImageJ (``ij.Macro``): a field's key is the first word of its label,
lower-cased, up to any ':' (``Macro.trimKey``); ``key=value`` pairs are separated by spaces;
values containing spaces are written ``key=[a value]``; a checkbox is on when its key appears
alone and **off when absent** (ImageJ macro mode); missing numeric/choice/string fields keep
their default.

``func(session, image, **values)`` does the work. Return values: ``None``; an
:class:`~rsstatchem.core.imaging.image.ImageDoc` (shown as a new image unless already open); a
:class:`~rsstatchem.core.imaging.results.ResultsTable` (shown as a table window); or a list of
those. Raise :class:`CommandError` for user errors (shown as an ImageJ-style message).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

KINDS = ("number", "slider", "checkbox", "choice", "radio", "string", "message", "file",
         "directory", "checkbox_group")


class CommandError(Exception):
    """A user-facing error ("There are no images open", "8-bit grayscale image required")."""


class Cancelled(Exception):
    """Raised by a command (or the progress hook) when the user cancels."""


def macro_key(label: str) -> str:
    """ImageJ ``Macro.trimKey``: first word, cut at ':', lower case."""
    key = label.strip().split(" ")[0]
    key = key.split(":")[0]
    return key.lower()


@dataclass
class Field:
    """One dialog field (GenericDialog ``addNumericField``, ``addCheckbox``, ...).

    kind: "number" | "slider" | "checkbox" | "choice" | "radio" | "string" | "message" |
    "file" | "directory" | "checkbox_group". ``items`` lists choices / radio buttons /
    checkbox-group labels (a checkbox group's default is a tuple of bools). ``decimals``
    (numbers) is also used to format recorded values; ``integer`` returns int.
    """
    kind: str
    label: str
    default: Any = None
    key: str | None = None
    items: tuple = ()
    decimals: int = 0
    units: str = ""
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    columns: int = 0          # string width / checkbox-group columns (layout hint)
    help: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown field kind {self.kind!r}")
        if self.key is None:
            self.key = macro_key(self.label)
        if self.default is None:
            self.default = {"number": 0.0, "slider": 0.0, "checkbox": False, "string": "",
                            "file": "", "directory": "", "message": "",
                            "checkbox_group": tuple(False for _ in self.items)}.get(
                self.kind, self.items[0] if self.items else "")

    @property
    def has_value(self) -> bool:
        return self.kind != "message"

    def keys(self) -> list[str]:
        """Macro keys used by this field (one per checkbox in a group)."""
        if self.kind == "checkbox_group":
            return [macro_key(item) for item in self.items]
        return [self.key] if self.has_value else []


# convenience constructors mirroring GenericDialog's add* methods
def number(label, default=0.0, decimals=0, units="", minimum=None, maximum=None, key=None,
           integer=False) -> Field:
    return Field("number", label, default, key, decimals=decimals, units=units,
                 minimum=minimum, maximum=maximum, integer=integer)


def slider(label, minimum, maximum, default, decimals=0, key=None) -> Field:
    return Field("slider", label, default, key, decimals=decimals, minimum=minimum,
                 maximum=maximum)


def checkbox(label, default=False, key=None) -> Field:
    return Field("checkbox", label, bool(default), key)


def choice(label, items, default=None, key=None) -> Field:
    items = tuple(items)
    return Field("choice", label, items[0] if default is None else default, key, items=items)


def radio(label, items, default=None, key=None, columns=0) -> Field:
    items = tuple(items)
    return Field("radio", label, items[0] if default is None else default, key, items=items,
                 columns=columns)


def string(label, default="", key=None, columns=0) -> Field:
    return Field("string", label, default, key, columns=columns)


def message(text) -> Field:
    return Field("message", text, "", key="")


def checkbox_group(labels, defaults=None, columns=0) -> Field:
    labels = tuple(labels)
    defaults = tuple(defaults) if defaults is not None else tuple(False for _ in labels)
    return Field("checkbox_group", "", defaults, key="", items=labels, columns=columns)


# ---------------------------------------------------------------------------------- options
_TOKEN = re.compile(r"\s*([^\s=\[]+)(?:=(\[[^\]]*\]|\S*))?")


def parse_options(options: str | None) -> dict[str, str | bool]:
    """``"sigma=2 stack title=[my image]"`` -> ``{"sigma": "2", "stack": True,
    "title": "my image"}`` (keys lower-cased as ImageJ compares them)."""
    out: dict[str, str | bool] = {}
    if not options:
        return out
    pos = 0
    text = options.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        key, value = m.group(1).lower(), m.group(2)
        if value is None:
            out.setdefault(key, True)
        else:
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            out.setdefault(key, value)
        pos = m.end()
    return out


def format_number(value: float, decimals: int = 0) -> str:
    """Numbers as ImageJ records them: integers without decimals, else ``decimals`` places
    (at least enough to not lose a non-integer value), no trailing exponent noise."""
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    d = max(decimals, 1)
    text = f"{v:.{d}f}"
    while float(text) != round(v, 9) and d < 9:
        d += 1
        text = f"{v:.{d}f}"
    return text


def _quote(value: str) -> str:
    return f"[{value}]" if (" " in value or value == "") else value


def values_from_options(fields: Iterable[Field], options: str | dict | None,
                        macro_mode: bool = True) -> dict[str, Any]:
    """Typed values for ``fields`` from an option string (or parsed dict).

    With ``macro_mode`` checkboxes absent from the string are False (ImageJ); otherwise the
    defaults are used for everything missing. Raises :class:`CommandError` for bad numbers
    or unknown choice items."""
    parsed = options if isinstance(options, dict) else parse_options(options)
    out: dict[str, Any] = {}
    for f in fields:
        if f.kind == "message":
            continue
        if f.kind == "checkbox_group":
            vals = []
            for item, default in zip(f.items, f.default):
                k = macro_key(item)
                vals.append(k in parsed if macro_mode else bool(parsed.get(k, default)))
            out[f.key or "group"] = tuple(vals)
            for item, v in zip(f.items, vals):
                out[macro_key(item)] = v
            continue
        raw = parsed.get(f.key)
        if f.kind == "checkbox":
            out[f.key] = (raw is not None) if macro_mode else (
                f.default if raw is None else raw is True or str(raw).lower() == "true")
            continue
        if raw is None or raw is True:
            out[f.key] = f.default
            continue
        if f.kind in ("number", "slider"):
            try:
                v = float(str(raw).replace("Infinity", "inf"))
            except ValueError:
                raise CommandError(f'Numeric value expected for "{f.key}": {raw}') from None
            out[f.key] = int(round(v)) if f.integer else v
        elif f.kind in ("choice", "radio"):
            if f.items and raw not in f.items:
                match = [i for i in f.items if i.lower() == str(raw).lower()]
                if not match:
                    raise CommandError(f'"{raw}" is not a valid choice for "{f.label}"')
                raw = match[0]
            out[f.key] = raw
        else:
            out[f.key] = str(raw)
    return out


def format_options(fields: Iterable[Field], values: dict[str, Any]) -> str:
    """Option string a recorder writes for ``values`` (field order; false checkboxes omitted)."""
    parts = []
    for f in fields:
        if f.kind == "message":
            continue
        if f.kind == "checkbox_group":
            for item in f.items:
                k = macro_key(item)
                if values.get(k):
                    parts.append(k)
            continue
        if f.key not in values:
            continue
        v = values[f.key]
        if f.kind == "checkbox":
            if v:
                parts.append(f.key)
        elif f.kind in ("number", "slider"):
            parts.append(f"{f.key}={format_number(v, f.decimals)}")
        else:
            parts.append(f"{f.key}={_quote(str(v))}")
    return " ".join(parts)


# ---------------------------------------------------------------------------------- commands
FieldsSpec = list | Callable[[Any, Any], list]


@dataclass
class Command:
    """A menu command. ``fields`` may be a callable ``(session, image) -> list[Field]`` when
    the dialog depends on the image (e.g. "Process all 27 slices?" or a channel list).

    ``ui_action`` (optional, set by UI ``commands_<area>`` modules) is a callable
    ``ui_action(bar, command)`` run instead of the generic dialog + ``func`` path for commands
    that open a window (Threshold..., ROI Manager..., B&C). Such commands may still provide a
    ``func`` for macros.
    """
    label: str
    func: Callable | None = None
    fields: FieldsSpec = field(default_factory=list)
    needs_image: bool = True
    types: tuple | None = None        # allowed ImageDoc.type values (None = all)
    supports_stack: bool = False      # ask "Process all N slices?" (adds the "stack" key)
    undoable: bool = True             # snapshot the image before running
    roi: str = "optional"             # "optional" | "required" | "area" | "line" | "none"
    recordable: bool = True
    owner: str = ""                   # area key from menu_layout.AREAS (filled by register)
    ui_action: Callable | None = None
    help: str = ""

    def dialog_fields(self, session=None, image=None) -> list[Field]:
        fields = self.fields(session, image) if callable(self.fields) else list(self.fields)
        if self.supports_stack and image is not None and image.stack_size > 1 and \
                not any(f.key == "stack" for f in fields):
            fields = fields + [checkbox(f"Process all {image.stack_size} slices", key="stack")]
        return fields

    @property
    def available(self) -> bool:
        return self.func is not None or self.ui_action is not None


class CommandRegistry:
    """Label -> :class:`Command`. One registry per Session (UI modules call
    ``register(registry)`` on it)."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command, replace: bool = False) -> Command:
        if command.label in self._commands and not replace:
            raise ValueError(f"command {command.label!r} is already registered")
        if not command.owner:
            from rsstatchem.core.imaging.menu_layout import owner_of
            command.owner = owner_of(command.label) or ""
        self._commands[command.label] = command
        return command

    def add(self, label: str, func: Callable | None = None, fields: FieldsSpec = (),
            replace: bool = False, **kw) -> Command:
        """Shorthand for ``register(Command(label, func, fields, **kw))``."""
        spec = fields if callable(fields) else list(fields)
        return self.register(Command(label, func, spec, **kw), replace=replace)

    def get(self, label: str) -> Command | None:
        return self._commands.get(label)

    def __contains__(self, label: str) -> bool:
        return label in self._commands

    def __iter__(self):
        return iter(self._commands.values())

    def __len__(self) -> int:
        return len(self._commands)

    def labels(self) -> list[str]:
        return list(self._commands)

    def find(self, text: str) -> list[Command]:
        """Find Commands: case-insensitive substring search over labels."""
        t = text.lower().strip()
        return [c for c in self._commands.values() if t in c.label.lower()]

    # ------------------------------------------------------------------ running
    def check(self, session, command: Command, image) -> None:
        """Raise CommandError when the command cannot run on ``image`` (ImageJ messages)."""
        if command.needs_image and image is None:
            raise CommandError("There are no images open")
        if image is not None and command.types is not None and image.type not in command.types:
            from rsstatchem.core.imaging.image import TYPE_NAMES
            names = " or ".join(TYPE_NAMES[t] for t in command.types)
            raise CommandError(f'"{command.label}" requires an image of type:\n \n{names}')
        if image is not None and command.roi in ("required", "area", "line"):
            roi = image.roi
            if roi is None or (command.roi == "area" and not roi.is_area()) or \
                    (command.roi == "line" and not roi.is_line()):
                kind = {"required": "selection", "area": "area selection",
                        "line": "line selection"}[command.roi]
                raise CommandError(f"{kind[0].upper()}{kind[1:]} required")

    def run(self, session, label: str, options: str | dict | None = "", image=None,
            values: dict | None = None, record: bool = True):
        """Run ``label`` headlessly (macro ``run(label, options)``).

        ``values`` (typed, e.g. from the Qt dialog) take precedence over ``options``. The
        command runs on ``image`` or the session's active image; returns what ``func``
        returned after the session has shown new images/tables."""
        command = self._commands.get(label)
        if command is None:
            raise CommandError(f'Unrecognized command: "{label}"')
        if command.func is None:
            raise CommandError(f'"{label}" is not available in batch mode')
        image = image if image is not None else session.active
        self.check(session, command, image)
        fields = command.dialog_fields(session, image)
        if values is None:
            values = values_from_options(fields, options)
        if command.needs_image and command.undoable and image is not None:
            image.snapshot(label)
        result = command.func(session, image, **values)
        if command.needs_image and image is not None and image in session.images:
            if command.undoable:
                image.changed = True
            session.image_updated(image)
        session.handle_result(result)
        session.last_command = (label, format_options(fields, values))
        if record and command.recordable:
            session.record(label, format_options(fields, values))
        return result


# ---------------------------------------------------------------------------------- recorder
class Recorder:
    """Plugins > Macros > Record: collects ``run(...)`` lines in ImageJ macro or Python form."""

    def __init__(self, language: str = "Macro") -> None:
        self.language = language     # "Macro" | "Python"
        self.lines: list[str] = []
        self.listeners: list[Callable[[str], None]] = []

    def record(self, label: str, options: str = "") -> str:
        line = macro_line(label, options) if self.language == "Macro" else \
            python_line(label, options)
        self.lines.append(line)
        for fn in list(self.listeners):
            fn(line)
        return line

    def record_line(self, line: str) -> None:
        """A non-``run`` statement (``open("...")``, ``selectImage("blobs.gif")``, ...)."""
        self.lines.append(line)
        for fn in list(self.listeners):
            fn(line)

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")

    def clear(self) -> None:
        self.lines.clear()


def _js_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macro_line(label: str, options: str = "") -> str:
    if options:
        return f"run({_js_string(label)}, {_js_string(options)});"
    return f"run({_js_string(label)});"


def python_line(label: str, options: str = "") -> str:
    """``ij.run("Gaussian Blur...", sigma=2, stack=True)`` (keywords that are identifiers),
    falling back to the option string otherwise."""
    parsed = parse_options(options)
    if parsed and all(k.isidentifier() for k in parsed):
        args = []
        for k, v in parsed.items():
            if v is True:
                args.append(f"{k}=True")
            else:
                try:
                    float(v)
                    args.append(f"{k}={v}")
                except ValueError:
                    args.append(f"{k}={v!r}")
        return f"ij.run({label!r}, {', '.join(args)})"
    return f"ij.run({label!r}, {options!r})" if options else f"ij.run({label!r})"
