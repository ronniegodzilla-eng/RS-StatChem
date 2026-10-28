"""Produce golden reference outputs for the Image Analysis module with the real ImageJ 1.54p.

ImageJ 1.x is public domain (NIH). This script downloads ``net.imagej:ij:1.54p`` from Maven
Central (SHA-256 pinned), prepares the inputs, runs every macro in ``tests/data_imagej/cases``
with the local JDK and writes compact goldens to ``tests/data_imagej/golden/<case>/``::

    python scripts/imagej_golden.py                 # all cases
    python scripts/imagej_golden.py --case filters  # one case (repeatable)
    python scripts/imagej_golden.py --inputs-only   # fetch/generate inputs, no ImageJ run

Layout of ``tests/data_imagej``:

* ``inputs/`` - public-domain files from the ImageJ repository's ``tests/data`` (pinned commit
  and SHA-256 below; never lena) and deterministic synthetic images made here (seeded numpy);
* ``cases/<case>.ijm`` - one macro per case; ``cases/_lib.ijm`` is prepended (``init()``,
  ``saveResults()``, ``saveTable()``, ``saveImage()``); the macro argument is "<in>|<out>";
* ``golden/<case>/`` - CSV files as the macro wrote them (9 decimals), and ``images.npz`` with
  every TIFF the macro saved (array name = file stem; read with ``numpy.load``, so tests need
  neither tifffile nor Java); ``golden/manifest.json`` records versions and checksums.

To add a case, drop ``cases/<name>.ijm`` and run ``--case <name>``; commit the inputs, the macro
and the new golden folder. pytest never runs Java. ImageJ is run through
``scripts/java/BatchImageJ.java`` (batch mode without ImageJ's main window), which needs a
desktop session: ImageJ's GenericDialog is an AWT dialog even when a macro supplies the values.
Requires a JDK >= 11 (``/usr/libexec/java_home`` on macOS, else ``java`` on PATH or $JAVA_HOME)
and tifffile.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data_imagej"
INPUTS, CASES, GOLDEN = DATA / "inputs", DATA / "cases", DATA / "golden"
CACHE = ROOT / "build" / "imagej"
DRIVER = ROOT / "scripts" / "java" / "BatchImageJ.java"

IJ_VERSION = "1.54p"
IJ_URL = f"https://repo1.maven.org/maven2/net/imagej/ij/{IJ_VERSION}/ij-{IJ_VERSION}.jar"
IJ_SHA256 = "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20"

# Public-domain inputs from github.com/imagej/ImageJ tests/data at this commit
IJ_REPO_COMMIT = "a376e43fad41a7337f7f01433c9673ed376b40e6"
PUBLIC_INPUTS = {
    "blobs.gif": "871f58484771ec6e23efcd2f1af3ca6ec1dc441d61ed59f743282fd76339aed3",
    "Cell_Colony.jpg": "d399005d4147a7a5e031743312e09e27f69bc398f2e99a9194a2606bca2fa780",
}
TIMEOUT = 600  # seconds per case (BatchImageJ exits on any ImageJ dialog)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, digest: str) -> Path:
    if dest.exists() and sha256(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out)
    got = sha256(part)
    if got != digest:
        part.unlink()
        raise SystemExit(f"SHA-256 mismatch for {url}: {got}")
    part.replace(dest)
    return dest


def find_java() -> str:
    home = os.environ.get("JAVA_HOME")
    if not home and sys.platform == "darwin" and Path("/usr/libexec/java_home").exists():
        home = subprocess.run(["/usr/libexec/java_home"], capture_output=True, text=True,
                              check=False).stdout.strip()
    if home and (Path(home) / "bin" / "java").exists():
        return str(Path(home) / "bin" / "java")
    java = shutil.which("java")
    if not java:
        raise SystemExit("no Java found (install a JDK >= 11 or set JAVA_HOME)")
    return java


# ---------------------------------------------------------------------------------- inputs
def synthetic_images() -> dict[str, np.ndarray]:
    """Deterministic test images (also stored in the goldens, so tests do not regenerate)."""
    rng = np.random.default_rng(20261027)
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    base = 40 + 1.2 * xx + 0.6 * yy
    for cx, cy, r, v in ((16, 18, 7, 120), (44, 20, 10, 90), (30, 46, 9, 140), (52, 50, 4, 60)):
        base += v * (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r)
    noisy = base + rng.normal(0, 12, base.shape)
    synth8 = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    synth16 = (synth8.astype(np.uint16) * 157 + 1000 +
               rng.integers(0, 157, synth8.shape)).astype(np.uint16)
    synth32 = (noisy / 37.0 - 2.5).astype(np.float32)
    # particles: discs, ellipses, a ring (hole), touching pairs and edge-touching objects
    yy, xx = np.mgrid[0:128, 0:160].astype(float)
    p = np.zeros(yy.shape, bool)
    for cx, cy, rx, ry in ((20, 20, 8, 8), (60, 22, 14, 6), (100, 25, 5, 12), (140, 20, 9, 9),
                           (25, 70, 3, 3), (70, 70, 12, 12), (88, 70, 7, 7), (130, 75, 15, 10),
                           (0, 110, 10, 10), (80, 127, 12, 6), (40, 110, 6, 9)):
        p |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    p &= ~(((xx - 70) ** 2 + (yy - 70) ** 2) <= 25)   # hole in the big disc
    p |= (np.abs(xx - 120) <= 1) & (yy >= 100) & (yy <= 120)   # thin bar
    particles8 = (p * 255).astype(np.uint8)
    return {"synth8": synth8, "synth16": synth16, "synth32": synth32,
            "particles8": particles8}


def prepare_inputs() -> dict[str, str]:
    import tifffile

    INPUTS.mkdir(parents=True, exist_ok=True)
    for name, digest in PUBLIC_INPUTS.items():
        url = (f"https://raw.githubusercontent.com/imagej/ImageJ/{IJ_REPO_COMMIT}/tests/data/"
               f"{name}")
        download(url, INPUTS / name, digest)
    for name, arr in synthetic_images().items():
        path = INPUTS / f"{name}.tif"
        tifffile.imwrite(path, arr, imagej=True, metadata={"axes": "YX"})
    return {p.name: sha256(p) for p in sorted(INPUTS.iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------------- cases
def run_case(name: str, java: str, jar: Path) -> list[str]:
    import tifffile

    macro = (CASES / "_lib.ijm").read_text() + "\n" + (CASES / f"{name}.ijm").read_text()
    out_dir = GOLDEN / name
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "out").mkdir()
        (tmp / "home").mkdir()
        mfile = tmp / f"{name}.ijm"
        mfile.write_text(macro)
        cmd = [java, "-Dapple.awt.UIElement=true", f"-Duser.home={tmp / 'home'}", "-cp",
               str(jar), str(DRIVER), str(mfile), f"{INPUTS}|{tmp / 'out'}"]
        print(f"running case {name} ...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, check=False)
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
            raise SystemExit(f"case {name} failed (exit {proc.returncode})")
        files = sorted((tmp / "out").iterdir())
        if not files:
            raise SystemExit(f"case {name} wrote nothing:\n{proc.stdout[-2000:]}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        arrays = {}
        written = []
        for f in files:
            if f.suffix == ".tif":
                arrays[f.stem] = tifffile.imread(f)
            else:
                shutil.copy2(f, out_dir / f.name)
                written.append(f.name)
        if arrays:
            if name == "filters":
                arrays.update({f"input_{k}": v for k, v in synthetic_images().items()
                               if k.startswith("synth")})
            np.savez_compressed(out_dir / "images.npz", **arrays)
            written.append("images.npz")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", action="append", help="case name (default: all)")
    ap.add_argument("--inputs-only", action="store_true")
    args = ap.parse_args(argv)
    input_sums = prepare_inputs()
    if args.inputs_only:
        return 0
    jar = download(IJ_URL, CACHE / f"ij-{IJ_VERSION}.jar", IJ_SHA256)
    java = find_java()
    version = subprocess.run([java, "-version"], capture_output=True, text=True,
                             check=False).stderr.splitlines()[0]
    names = args.case or sorted(p.stem for p in CASES.glob("*.ijm") if not p.stem.startswith("_"))
    manifest_path = GOLDEN / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"imagej": IJ_VERSION, "ij_jar_sha256": IJ_SHA256, "inputs": input_sums,
                     "imagej_repo_commit": IJ_REPO_COMMIT})
    cases = manifest.setdefault("cases", {})
    for name in names:
        files = run_case(name, java, jar)
        cases[name] = {"files": files, "java": version,
                       "macro_sha256": sha256(CASES / f"{name}.ijm"),
                       "date": _dt.date.today().isoformat()}
    GOLDEN.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"golden files written to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
