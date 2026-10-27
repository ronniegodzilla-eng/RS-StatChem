"""Download the pinned Chemistry Development Kit (CDK) module jars and build RS-StatChem's small
CDK helper (open logP / molar-refractivity methods for the Chemical Properties window).

CDK (https://cdk.github.io; C. Steinbeck et al., J. Chem. Inf. Comput. Sci. 43 (2003) 493;
E. L. Willighagen et al., J. Cheminform. 9 (2017) 33) is licensed LGPL-2.1-or-later. The app
runs it unmodified, as a separate Java process (``rsstatchem.core.chem_draw.cdk``), with the
same Java runtime as OPSIN (``resources/opsin/jre`` in the packaged app, else a system Java).
Neither the jars nor the compiled helper are committed to git; this script fetches the jars
from Maven Central, verifies each SHA-256 and compiles ``scripts/java/CdkProps.java`` with
``javac`` (JDK >= 11)::

    python scripts/fetch_cdk.py      # -> rsstatchem/resources/java/{cdk/*.jar,
                                     #    rsstatchem-cdk.jar, cdk-classpath.txt, VERSION,
                                     #    LICENSE-CDK.txt}

Only the CDK modules the helper actually loads are fetched (13 unmodified jars, 1.9 MB instead
of the 24 MB ``cdk-2.13.jar`` bundle): the list comes from ``java -Xlog:class+load`` over the
whole logP benchmark set plus every structure of the experimental database (53 000 SMILES),
and the helper gives byte-identical answers for all of them with these jars and with the full
bundle (``--compare-bundle`` repeats that check). The helper is started with an explicit
class path (``cdk-classpath.txt``, in this order).

Java modules: the helper only parses SMILES and runs the JPlogP / XLogP / ALOGP descriptors,
which need no more than the modules OPSIN already uses (``java.base,java.xml``) - verified
by running it over the whole logP benchmark set with such a runtime (``--check-runtime``
repeats a smaller check). CDK logs through its own stderr logger here (no log4j on the class
path); the log4j property in ``cdk.CDK_LOGGING`` is harmless.
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
MAVEN = "https://repo1.maven.org/maven2"
# (groupId, artifactId, version, SHA-256 of the jar on Maven Central), class-path order
JARS = (
    ("org.openscience.cdk", "cdk-interfaces", VERSION,
     "b0c92214fdfa75785fe92a2e3e15d590ff31f7a9b73f64f12ce428b67c90478b"),
    ("org.openscience.cdk", "cdk-core", VERSION,
     "e1a2961885426d91e29c7c6b86804b30adeccec53482613de4410112290907b7"),
    ("org.openscience.cdk", "cdk-standard", VERSION,
     "7b1e955946417ee3402122aa0067699e177404218fa5ab9cc659476c3b484605"),
    ("org.openscience.cdk", "cdk-silent", VERSION,
     "c4eda4270a2fd14bfae2ec27c227594eb6585233689c3537f4f0826c323ee138"),
    ("org.openscience.cdk", "cdk-valencycheck", VERSION,
     "83764773eb30ac1fea638135d03a16d1bcf3c49a109264037fcb07affcd20d1a"),
    ("org.openscience.cdk", "cdk-atomtype", VERSION,
     "1d62b30bbaede60c551067a7269bd90f868b4b62306bfd648c1d5b01b46ab1e5"),
    ("org.openscience.cdk", "cdk-isomorphism", VERSION,
     "f3bc7716dae284a4d3b4a35b8a09930d2c2e65226f35e7ee1a8a753a49750d35"),
    ("org.openscience.cdk", "cdk-smarts", VERSION,
     "4c4dab86c92122577f5bfa4363c5883b50526e33190f5fa30b829cb847bec19c"),
    ("org.openscience.cdk", "cdk-smiles", VERSION,
     "f2124e65f65f959f8f240897303eaa69b8ffe2a9b8b02eb53afbee7fffb73072"),
    ("org.openscience.cdk", "cdk-qsar", VERSION,
     "111a6c0a48153e54abeae641cca3abe0aad557cd493207fd0f38f020675f62af"),
    ("org.openscience.cdk", "cdk-qsarmolecular", VERSION,
     "1ec145d49d2f76e3033ad4bfd70b42f0688594f84f6831e578ed104474ad0c3c"),
    ("uk.ac.ebi.beam", "beam-core", "1.3.10",
     "f097648ebac03eda708aa7b3909900b37d19e1323843de071d7e8316f9162b57"),
    ("javax.vecmath", "vecmath", "1.5.2",
     "3558e81ca74c60dc01baca7ef05f48594bbaed43af45d4f1b9196e479734e675"),
)
# the former single-jar bundle, only for --compare-bundle
BUNDLE_URL = f"https://github.com/cdk/cdk/releases/download/cdk-{VERSION}/cdk-{VERSION}.jar"
BUNDLE_SHA256 = "5a42221d412dca53e48fdfc0bfae645310f9de80082800af801d06a87df32411"
JLINK_MODULES = "java.base,java.xml"
CLASSPATH_FILE = "cdk-classpath.txt"


def jar_name(artifact: str, version: str) -> str:
    return f"{artifact}-{version}.jar"


def jar_url(group: str, artifact: str, version: str) -> str:
    return f"{MAVEN}/{group.replace('.', '/')}/{artifact}/{version}/{jar_name(artifact, version)}"


def license_text() -> str:
    rows = "\n".join(f"  {jar_name(a, v):28s} sha256 {h}\n      {jar_url(g, a, v)}"
                     for g, a, v, h in JARS)
    return f"""The Chemistry Development Kit (CDK) {VERSION}
https://cdk.github.io  -  source: https://github.com/cdk/cdk/tree/cdk-{VERSION}

The CDK is free software; you can redistribute it and/or modify it under the terms of the GNU
Lesser General Public License as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version. A copy of the licence is available at
https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html .

RS-StatChem uses these unmodified jars from Maven Central (folder cdk/), as a separate
program: rsstatchem-cdk.jar (source: scripts/java/CdkProps.java in the RS-StatChem source)
talks to them over standard input/output and is started with the class path listed in
{CLASSPATH_FILE}. You may replace any of these jars with another (e.g. modified) build of a
compatible version. Sources of every CDK module: the tag cdk-{VERSION} above, or the
*-sources.jar next to each jar on Maven Central.

{rows}

Third-party libraries among them, under their own licences:

* vecmath 1.5.2 (javax.vecmath, 3D vector maths): GNU GPL version 2 with the Classpath
  exception (https://openjdk.org/legal/gplv2+ce.html); source: vecmath-1.5.2-sources.jar
  next to the jar on Maven Central, https://github.com/hharrison/vecmath .
* beam-core 1.3.10 (SMILES parser, John May; https://github.com/johnmay/beam), BSD 2-Clause:

{BEAM_LICENSE}"""


BEAM_LICENSE = """Copyright (c) 2013, European Bioinformatics Institute (EMBL-EBI)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: Path, digest: str) -> None:
    tmp = target.with_suffix(".part")
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "RS-StatChem build"})
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
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        raise SystemExit(f"SHA-256 mismatch for {target.name}: got {got}, expected {digest}")
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


def lib_jars(dest: Path = DEST) -> list[Path]:
    return [dest / "cdk" / jar_name(a, v) for _g, a, v, _h in JARS]


def build_helper(dest: Path) -> Path:
    """Compile CdkProps.java against the CDK jars into rsstatchem-cdk.jar (Java 11 bytecode)."""
    jar = dest / "rsstatchem-cdk.jar"
    cp = os.pathsep.join(str(j) for j in lib_jars(dest))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([_jdk_tool("javac"), "--release", "11", "-Xlint:-options",
                        "-nowarn", "-encoding", "UTF-8", "-cp", cp, "-d", tmp, str(SOURCE)],
                       check=True)
        subprocess.run([_jdk_tool("jar"), "--create", "--file", str(jar), "-C", tmp, "."],
                       check=True)
    print(f"Built {jar} ({jar.stat().st_size / 1e3:.1f} kB)")
    return jar


def fetch(dest: Path = DEST, force: bool = False, build: bool = True) -> list[Path]:
    lib = dest / "cdk"
    lib.mkdir(parents=True, exist_ok=True)
    wanted: set[str] = set()
    for g, a, v, digest in JARS:
        target = lib / jar_name(a, v)
        wanted.add(target.name)
        if target.exists() and not force and sha256(target) == digest:
            continue
        _download(jar_url(g, a, v), target, digest)
    # stale jars: other versions in cdk/ and the former single-jar bundle resources/java/cdk.jar
    stale = [p for p in lib.glob("*.jar") if p.name not in wanted] + \
        [p for p in (dest / "cdk.jar",) if p.exists()]
    for old in stale:
        old.unlink()
        print(f"Removed {old}")
    total = sum(p.stat().st_size for p in lib_jars(dest))
    print(f"{len(JARS)} CDK jars in {lib} ({total / 1e6:.2f} MB, SHA-256 verified)")
    (dest / CLASSPATH_FILE).write_text(
        "\n".join(f"cdk/{jar_name(a, v)}" for _g, a, v, _h in JARS) + "\n", encoding="utf-8")
    (dest / "VERSION").write_text(
        f"CDK {VERSION} (Maven Central module jars)\n" +
        "".join(f"{jar_name(a, v)} sha256 {h}\n" for _g, a, v, h in JARS), encoding="utf-8")
    (dest / "LICENSE-CDK.txt").write_text(license_text(), encoding="utf-8")
    if build:
        build_helper(dest)
    return lib_jars(dest)


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


def compare_bundle(smiles_file: Path, java: str | None, dest: Path = DEST) -> int:
    """Answer every line of ``smiles_file`` (``id<TAB>SMILES``) with the module jars and with
    the former single cdk-2.13.jar bundle (downloaded to a temporary folder, SHA-256 checked)
    and report whether the outputs are byte-identical."""
    java = java or _jdk_tool("java")
    data = smiles_file.read_bytes().rstrip(b"\n") + b"\nquit\n"
    helper = dest / "rsstatchem-cdk.jar"
    base = [java, "-Dfile.encoding=UTF-8", "-Xss8m", "-XX:+UseSerialGC", "-cp"]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / f"cdk-{VERSION}.jar"
        _download(BUNDLE_URL, bundle, BUNDLE_SHA256)
        outs = []
        for cp in ([helper] + lib_jars(dest), [helper, bundle]):
            r = subprocess.run(base + [os.pathsep.join(map(str, cp)), "CdkProps"], input=data,
                               capture_output=True, check=True)
            outs.append(r.stdout)
    n = outs[0].count(b"\n") - 1
    same = outs[0] == outs[1]
    print(f"{n} molecules: module jars and full bundle give "
          f"{'identical' if same else 'DIFFERENT'} answers")
    return 0 if same else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch CDK (SHA-256 verified) and build the helper")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="download even if present")
    ap.add_argument("--no-build", action="store_true", help="do not compile the helper")
    ap.add_argument("--print-modules", action="store_true",
                    help="print the jlink module list and exit")
    ap.add_argument("--check-runtime", metavar="JAVA",
                    help="run the helper with this java executable and exit")
    ap.add_argument("--compare-bundle", metavar="SMILES_FILE", type=Path,
                    help="compare the module jars with the full CDK bundle on id<TAB>SMILES "
                         "lines and exit")
    ap.add_argument("--java", help="java executable for --compare-bundle")
    args = ap.parse_args(argv)
    if args.print_modules:
        print(JLINK_MODULES)
        return 0
    if args.check_runtime:
        return check_runtime(args.check_runtime)
    if args.compare_bundle:
        return compare_bundle(args.compare_bundle, args.java, args.dest)
    fetch(args.dest, args.force, build=not args.no_build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
