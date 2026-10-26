"""Download the pinned OPSIN command-line jar (offline name -> structure) into the app resources.

OPSIN (Open Parser for Systematic IUPAC Nomenclature, D. M. Lowe et al., J. Chem. Inf. Model.
51 (2011) 739; https://github.com/dan2097/opsin) is MIT licensed. RS-StatChem runs it as a
separate Java process (``rsstatchem.core.chem_draw.names.opsin_structure``), so neither the
jar nor a Java runtime is committed to git; this script fetches the jar and verifies its
SHA-256, and the build workflow adds a minimal Java runtime made with ``jlink``.

Run from the project root::

    python scripts/fetch_opsin.py            # -> rsstatchem/resources/opsin/opsin-cli.jar

Minimal Java runtime for packaging (done by .github/workflows/build.yml; any JDK >= 11,
the workflow pins Temurin 21)::

    jdeps --print-module-deps --ignore-missing-deps rsstatchem/resources/opsin/opsin-cli.jar
    # lists log4j-core extras (java.desktop, java.naming...); RS-StatChem starts OPSIN with
    # log4j's simple logger (names.OPSIN_LOGGING), so java.base + java.xml are enough:
    jlink --add-modules java.base,java.xml --strip-debug --no-man-pages \\
          --no-header-files --compress=zip-9 --output rsstatchem/resources/opsin/jre

(``--compress=2`` instead of ``zip-9`` on JDK < 21.) The resulting runtime is about 30-40 MB
on disk (35 MB measured on macOS arm64). For a local test, ``--output`` may point at
``rsstatchem/resources/opsin/jre`` (git-ignored); the packaging spec leaves that folder out and
CI copies the runtime into the finished bundle instead. The app looks for
``resources/opsin/jre/bin/java`` (``java.exe`` on Windows) first, then for a system ``java``
(``JAVA_HOME`` or ``PATH``).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "rsstatchem" / "resources" / "opsin"
VERSION = "2.9.0"
JAR = f"opsin-cli-{VERSION}-jar-with-dependencies.jar"
URL = f"https://github.com/dan2097/opsin/releases/download/{VERSION}/{JAR}"
SHA256 = "c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f"
JLINK_MODULES = "java.base,java.xml"
LICENSE = f"""OPSIN {VERSION} - Open Parser for Systematic IUPAC Nomenclature
https://github.com/dan2097/opsin  (MIT License)

Copyright (c) 2009-2026 Daniel Lowe and the OPSIN contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path = DEST, force: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "opsin-cli.jar"
    if target.exists() and not force and sha256(target) == SHA256:
        print(f"{target} is already OPSIN {VERSION} (SHA-256 verified)")
        return target
    tmp = target.with_suffix(".part")
    print(f"Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "RS-StatChem build"})
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        raise SystemExit(f"SHA-256 mismatch for {JAR}: got {digest}, expected {SHA256}")
    tmp.replace(target)
    (dest / "VERSION").write_text(f"OPSIN {VERSION}\n{URL}\nsha256 {SHA256}\n",
                                  encoding="utf-8")
    (dest / "LICENSE-OPSIN.txt").write_text(LICENSE, encoding="utf-8")
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB, SHA-256 verified)")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the pinned OPSIN jar (SHA-256 verified)")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="download even if present")
    ap.add_argument("--print-modules", action="store_true",
                    help="print the jlink module list and exit")
    args = ap.parse_args(argv)
    if args.print_modules:
        print(JLINK_MODULES)
        return 0
    fetch(args.dest, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
