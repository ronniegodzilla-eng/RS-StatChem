"""The PDF user guide stays complete: every menu command is explained, every picture exists.

The full build (screenshots + PDF) runs in CI (``docs/user_guide/build.py --strict``); these
checks are the fast part, so a new menu command without an explanation is caught locally.
"""
import re
import sys
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "user_guide"
sys.path.insert(0, str(GUIDE))

from guide import features as F  # noqa: E402
from guide import markup  # noqa: E402
import menu_states  # noqa: E402,F401 - registers the extra module states

MODULES = ("stats", "rsm", "graph", "chem", "image")


def _chapters():
    return {p.name: markup.parse_file(p) for p in sorted((GUIDE / "content").glob("*.md"))}


def _scene_keys():
    keys = set()
    for path in (GUIDE / "scenes").glob("*.py"):
        keys |= set(re.findall(r'@scene\(\s*"([\w.\-]+)"', path.read_text(encoding="utf-8")))
    return keys


def test_chapters_parse_and_pictures_have_scenes():
    keys = _scene_keys()
    missing = []
    for name, blocks in _chapters().items():
        assert blocks and blocks[0].kind == "h1", f"{name} must start with a # chapter title"
        for b in blocks:
            if b.kind == "figure":
                head, _, tail = b.arg.rpartition(".")
                if b.arg not in keys and not (tail.isdigit() and head in keys):
                    missing.append(f"{b.source}: {b.arg}")
    assert missing == []


def test_every_module_chapter_has_its_reference():
    used = {b.arg for blocks in _chapters().values() for b in blocks if b.kind == "features"}
    assert used == set(MODULES)


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    yield context
    for w in list(context.windows):
        w.hide()


@pytest.mark.parametrize("module", MODULES)
def test_every_menu_command_is_explained(ctx, module):
    from rsstatchem.ui.modules import BY_KEY, open_module

    window = open_module(ctx, module)
    sections = F.read_all(window, module, BY_KEY[module].title)
    assert len(sections[0].menus.commands) > 20
    missing = F.missing(sections)
    assert missing == [], (f"add these to docs/user_guide/features/{module}.tsv: "
                           + "; ".join(missing))
    assert F.unused(sections) == [], "explanations for commands that no longer exist"
