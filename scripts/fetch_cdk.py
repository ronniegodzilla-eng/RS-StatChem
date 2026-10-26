"""Download the pinned Chemistry Development Kit (CDK) jar and build RS-StatChem's small CDK
helper (open logP / molar-refractivity methods for the Chemical Properties window).

CDK (https://cdk.github.io; C. Steinbeck et al., J. Chem. Inf. Comput. Sci. 43 (2003) 493;
E. L. Willighagen et al., J. Cheminform. 9 (2017) 33) is licensed LGPL-2.1-or-later. The app
runs it unmodified, as a separate Java process (``rsstatchem.core.chem_draw.cdk``), with the
same Java runtime as OPSIN (``resources/opsin/jre`` in the packaged app, else a system Java).
Neither the jar nor the compiled helper is committed to git; this script fetches the jar,
verifies its SHA-256 and compiles ``scripts/java/CdkProps.java`` with ``javac`` (JDK >= 11)::

    python scripts/fetch_cdk.py            # -> rsstatchem/resources/java/{cdk.jar,
                                           #    rsstatchem-cdk.jar, VERSION, LICENSE-CDK.txt}

Java modules: ``jdeps`` on the whole CDK bundle lists java.desktop, java.sql, java.naming ...
(rendering, I/O and logging back-ends RS-StatChem never loads). The helper only parses SMILES
and runs the JPlogP / XLogP / ALOGP descriptors, which need no more than the modules OPSIN
already uses (``java.base,java.xml``) - verified by running it over the whole logP benchmark
set with such a runtime (``--check-runtime`` repeats a smaller check). Log4j is pointed at its
simple logger (``CDK_LOGGING``) so that no log4j-core module requirements apply.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "rsstatchem" / "resources" / "java"
SOURCE = ROOT / "scripts" / "java" / "CdkProps.java"
VERSION = "2.13"
URL = f"https://github.com/cdk/cdk/releases/download/cdk-{VERSION}/cdk-{VERSION}.jar"
SHA256 = "5a42221d412dca53e48fdfc0bfae645310f9de80082800af801d06a87df32411"
JLINK_MODULES = "java.base,java.xml"
LICENSE = f"""The Chemistry Development Kit (CDK) {VERSION}
https://cdk.github.io  -  source: https://github.com/cdk/cdk/tree/cdk-{VERSION}

The CDK is free software; you can redistribute it and/or modify it under the terms of the GNU
Lesser General Public License as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version. A copy of the licence is available at
https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html .

RS-StatChem uses the unmodified CDK release jar (cdk-{VERSION}.jar, sha256 {SHA256})
as a separate program: rsstatchem-cdk.jar (source: scripts/java/CdkProps.java in the
RS-StatChem source) talks to it over standard input/output. You may replace cdk.jar with
another (e.g. modified) CDK build of a compatible version.

The CDK bundle contains third-party libraries under their own licences (see the jar's
META-INF and license folders and the CDK release notes): among them Beam, Apache Commons, Apache
Log4j, Joda-Time, vecmath, CMLXOM/Euclid, FreeHEP, JNA-InChI and gilleain signatures.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(target: Path) -> None:
    tmp = target.with_suffix(".part")
    print(f"Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "RS-StatChem build"})
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        raise SystemExit(f"SHA-256 mismatch for cdk-{VERSION}.jar: got {digest}, "
                         f"expected {SHA256}")
    tmp.replace(target)


def _jdk_tool(name: str) -> str:
    home = os.environ.get("JAVA_HOME", "")
    exe = name + (".exe" if sys.platform == "win32" else "")
    if home and (Path(home) / "bin" / exe).is_file():
        return str(Path(home) / "bin" / exe)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"'{name}' not found: install a JDK (>= 11) or set JAVA_HOME")
    return found


def build_helper(dest: Path) -> Path:
    """Compile CdkProps.java against cdk.jar into rsstatchem-cdk.jar (Java 11 bytecode)."""
    jar = dest / "rsstatchem-cdk.jar"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([_jdk_tool("javac"), "--release", "11", "-Xlint:-options",
                        "-nowarn", "-encoding", "UTF-8", "-cp", str(dest / "cdk.jar"),
                        "-d", tmp, str(SOURCE)], check=True)
        subprocess.run([_jdk_tool("jar"), "--create", "--file", str(jar), "-C", tmp, "."],
                       check=True)
    print(f"Built {jar} ({jar.stat().st_size / 1e3:.1f} kB)")
    return jar


def fetch(dest: Path = DEST, force: bool = False, build: bool = True) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "cdk.jar"
    if target.exists() and not force and sha256(target) == SHA256:
        print(f"{target} is already CDK {VERSION} (SHA-256 verified)")
    else:
        _download(target)
        print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB, SHA-256 verified)")
    (dest / "VERSION").write_text(f"CDK {VERSION}\n{URL}\nsha256 {SHA256}\n", encoding="utf-8")
    (dest / "LICENSE-CDK.txt").write_text(LICENSE, encoding="utf-8")
    if build:
        build_helper(dest)
    return target


def check_runtime(java: str) -> int:
    """Run the helper with the given java on a few molecules (for a jlink runtime)."""
    sys.path.insert(0, str(ROOT))
    from rsstatchem.core.chem_draw import cdk
    helper = cdk.CdkHelper(java=java)
    try:
        for smi in ("c1ccccc1", "CCO", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O",
                    "OC(=O)c1ccccc1O", "ClC(Cl)(Cl)Cl", "C1CCC2(CC1)OCCO2"):
            res = helper.compute(smi, timeout=120)
            print(smi, res)
            if res.get("jplogp") is None or res.get("xlogp") is None:
                return 1
    finally:
        helper.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch CDK (SHA-256 verified) and build the helper")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="download even if present")
    ap.add_argument("--no-build", action="store_true", help="do not compile the helper")
    ap.add_argument("--print-modules", action="store_true",
                    help="print the jlink module list and exit")
    ap.add_argument("--check-runtime", metavar="JAVA",
                    help="run the helper with this java executable and exit")
    args = ap.parse_args(argv)
    if args.print_modules:
        print(JLINK_MODULES)
        return 0
    if args.check_runtime:
        return check_runtime(args.check_runtime)
    fetch(args.dest, args.force, build=not args.no_build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
