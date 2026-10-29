"""Build the RS-StatChem user guide (PDF) from the running app.

    python docs/user_guide/build.py [--out RS-StatChem-User-Guide.pdf] [--only stats]
                                    [--strict] [--shots DIR]

1. every chapter in ``content/`` (in file-name order) is parsed;
2. the screenshot scenes in ``scenes/`` that the chapters use are captured offscreen;
3. each module's menus are read for the feature reference (texts in ``features/``);
4. the PDF is laid out.

``--strict`` fails when a menu command has no explanation or a scene is missing, which is how
CI keeps the guide complete when features are added. ``--only`` limits the chapters (and their
scenes) to the given content file prefixes or module keys, for quick previews.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from guide import features as F  # noqa: E402
from guide import markup  # noqa: E402
from guide.studio import SCENES, Studio  # noqa: E402  (sets up the offscreen platform)

CONTENT = HERE / "content"
SCENE_DIR = HERE / "scenes"
MODULE_KEYS = ("stats", "rsm", "graph", "chem", "image")


def load_scenes() -> None:
    for path in sorted(SCENE_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"guide_scenes_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def chapters(only: list[str] | None) -> list[tuple[Path, list[markup.Block]]]:
    out = []
    for path in sorted(CONTENT.glob("*.md")):
        if only and not any(path.stem.startswith(o) or o in path.stem for o in only):
            continue
        out.append((path, markup.parse_file(path)))
    return out


def capture(st: Studio, keys: list[str], shots: Path, strict: bool):
    from guide.pdf import Figure

    figures = {}
    problems = []
    for key in keys:
        info = SCENES.get(key)
        if info is None:
            problems.append(f"scene {key!r} is used but not defined in scenes/")
            continue
        t0 = time.perf_counter()
        try:
            result = info.func(st)
        except Exception as exc:  # noqa: BLE001 - report every broken scene at once
            import traceback

            problems.append(f"scene {key!r} failed: {exc}\n{traceback.format_exc()}")
            st.close_all()
            continue
        shots_list = result if isinstance(result, list) else [result]
        for i, shot in enumerate(shots_list):
            name = key if i == 0 else f"{key}.{i + 1}"
            path = shots / f"{name}.png"
            legend = st.render(shot, path)
            figures[name] = Figure(path, info.caption, legend)
        st.close_all()
        print(f"  {key:<40} {time.perf_counter() - t0:5.1f}s")
    return figures, problems


def menu_references(st: Studio, modules: list[str]):
    import menu_states  # noqa: F401 - registers the extra module states
    from rsstatchem.ui.modules import BY_KEY

    refs, problems = {}, []
    for key in modules:
        info = BY_KEY[key]
        window = st.module(key)
        sections = F.read_all(window, key, info.title)
        refs[key] = (info.title, [(s.label, F.rows(s.menus, s.label)) for s in sections])
        miss = F.missing(sections)
        stale = F.unused(sections)
        if miss:
            problems.append(f"{key}: {len(miss)} menu commands without an explanation in "
                            f"features/{key}.tsv, e.g. " + "; ".join(miss[:8]))
        if stale:
            print(f"  note: {key}: explanations for commands that no longer exist: "
                  + "; ".join(stale[:8]))
        n = sum(len(s.menus.commands) for s in sections)
        print(f"  menus {key:<6} {n:4d} commands in {len(sections)} section(s), "
              f"{len(miss)} without text")
        st.close_all()
    return refs, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "dist" / "RS-StatChem-User-Guide.pdf"))
    ap.add_argument("--only", nargs="*", help="content file prefixes / module keys")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--shots", default=str(ROOT / "build" / "user_guide" / "shots"))
    args = ap.parse_args(argv)

    from rsstatchem import __version__
    from guide.pdf import Builder, date_text

    t0 = time.perf_counter()
    load_scenes()
    chs = chapters(args.only)
    keys, modules = [], []
    for _, blocks in chs:
        for b in blocks:
            if b.kind == "figure" and b.arg not in keys:
                keys.append(b.arg)
            if b.kind == "features" and b.arg not in modules:
                modules.append(b.arg)
    cover_key = "intro.launcher"
    if cover_key not in keys and cover_key in SCENES:
        keys.insert(0, cover_key)
    # a multi-picture scene "a.b" is used as "a.b.2" and so on
    base_keys = []
    for k in keys:
        head, _, tail = k.rpartition(".")
        base = head if k not in SCENES and tail.isdigit() and head in SCENES else k
        if base not in base_keys:
            base_keys.append(base)

    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    print(f"RS-StatChem {__version__}: {len(chs)} chapters, {len(base_keys)} scenes, "
          f"{len(modules)} menu references")
    st = Studio()
    figures, problems = capture(st, base_keys, shots, args.strict)
    refs, menu_problems = menu_references(st, modules)
    problems += menu_problems
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if problems and args.strict:
        return 1
    if any(p.startswith("scene") for p in problems):
        # still lay out the rest, with the broken pictures left out
        for _, blocks in chs:
            blocks[:] = [b for b in blocks if not (b.kind == "figure" and b.arg not in figures)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cover = figures.get(cover_key)
    Builder(figures, refs, __version__, date_text(dt.date.today()),
            cover.path if cover else None).build([b for _, b in chs], out)
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB) in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
