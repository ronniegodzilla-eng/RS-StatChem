"""Debugging aid: find slots that would take Qt's ``checked`` flag for a real argument.

    python scripts/slot_audit.py

``QAction.triggered`` and ``QAbstractButton.clicked`` carry a bool. PySide passes it to a slot
whenever the slot can take a positional argument, so ``def save_as(self, path=None)`` connected
directly receives ``path=False``. This lists every method that is handed over un-wrapped
(``x.triggered.connect(self.m)``, ``addAction(text, self.m)``, an action factory...) and whose
first parameter has a non-bool default. Wrap such a slot: ``lambda: self.m()``.

Exit code 1 when something is found (the test suite runs this).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOL_SIGNALS = {"triggered", "clicked", "toggled"}
# parameter names that are meant to receive the flag
FLAG_NAMES = {"checked", "on", "state", "enabled", "flag", "value", "visible", "show", "_",
              "_checked", "_on", "b", "yes", "active", "toggled", "check"}


def first_param(fn: ast.FunctionDef, bound: bool):
    args = list(fn.args.posonlyargs) + list(fn.args.args)
    defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
    if bound and args and args[0].arg in ("self", "cls"):
        args, defaults = args[1:], defaults[1:]
    if not args:
        return None
    return args[0], defaults[0]


def risky(fn: ast.FunctionDef, bound: bool) -> str | None:
    """Why ``fn`` must not get the checked flag, or None."""
    fp = first_param(fn, bound)
    if fp is None:
        return None
    arg, default = fp
    if default is None:
        return None                      # required parameter: written to take the flag
    if arg.arg in FLAG_NAMES:
        return None
    ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
    if ann == "bool" or (isinstance(default, ast.Constant) and isinstance(default.value, bool)):
        return None
    return f"{arg.arg}{': ' + ann if ann else ''} = {ast.unparse(default)}"


def direct_slots(cls: ast.ClassDef):
    """(line, method name, how) for every un-wrapped ``self.m`` handed to a bool signal inside
    ``cls``. The module windows' own action helpers are safe (rsstatchem.ui.actions)."""
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        how = None
        if isinstance(func, ast.Attribute) and func.attr == "connect" and \
                isinstance(func.value, ast.Attribute) and func.value.attr in BOOL_SIGNALS:
            how = f".{func.value.attr}.connect"
        elif isinstance(func, ast.Attribute) and func.attr in ("addAction", "addButton"):
            how = f".{func.attr}"
        if how is None:
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) \
                    and arg.value.id == "self":
                yield node.lineno, arg.attr, how


def audit() -> list[str]:
    classes: dict[str, list[tuple[Path, ast.ClassDef]]] = {}
    for path in sorted((ROOT / "rsstatchem").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            classes.setdefault(cls.name, []).append((path, cls))

    def bases(cls: ast.ClassDef) -> list[str]:
        return [b.id if isinstance(b, ast.Name) else b.attr for b in cls.bases
                if isinstance(b, (ast.Name, ast.Attribute))]

    def family(name: str) -> list[tuple[Path, ast.ClassDef]]:
        """The class, its ancestors and every class derived from it (by name)."""
        out, todo, seen = [], [name], set()
        while todo:                                  # ancestors
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            for path, cls in classes.get(n, []):
                out.append((path, cls))
                todo += bases(cls)
        changed = True
        down = {name}
        while changed:                               # descendants (overrides)
            changed = False
            for n, defs in classes.items():
                for path, cls in defs:
                    if n not in down and down & set(bases(cls)):
                        down.add(n)
                        out.append((path, cls))
                        changed = True
        return out

    findings = []
    for name, defs in classes.items():
        for path, cls in defs:
            for line, method, how in direct_slots(cls):
                for def_path, owner in family(name):
                    for item in owner.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method:
                            why = risky(item, bound=True)
                            if why:
                                findings.append(
                                    f"{path.relative_to(ROOT)}:{line}: self.{method} via {how}"
                                    f" -> {def_path.relative_to(ROOT)}:{item.lineno} "
                                    f"{owner.name}.{method}(self, {why})")
    return sorted(set(findings))


if __name__ == "__main__":
    found = audit()
    print("\n".join(found) if found else "no slot takes the checked flag for an argument")
    sys.exit(1 if found else 0)
