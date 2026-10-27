"""Open logP and molar-refractivity methods from the Chemistry Development Kit (CDK).

CDK (https://cdk.github.io, LGPL-2.1) implements several published logP methods; RS-StatChem
runs four of them through a small helper (``scripts/java/CdkProps.java`` compiled into
``resources/java/rsstatchem-cdk.jar`` next to the unmodified CDK module jars in
``resources/java/cdk/`` (class path in ``resources/java/cdk-classpath.txt``) by
``scripts/fetch_cdk.py``):

- JPlogP - J. Plante, S. Werner, J. Cheminform. 10 (2018) 61;
- XLogP (v2) - R. Wang, Y. Fu, L. Lai, J. Chem. Inf. Comput. Sci. 37 (1997) 615;
- ALogP and AMR (molar refractivity) - A. K. Ghose, V. N. Viswanadhan, J. J. Wendoloski,
  J. Phys. Chem. A 102 (1998) 3762.

These are open-source alternatives, *not* BioByte's CLogP / CMR; values differ.

The helper is **one long-lived Java process** (started lazily on first use, restarted when it
dies or stops answering) that reads ``id<TAB>SMILES`` lines on stdin and answers one JSON line
each, so the JVM starts only once. It uses the same Java runtime as OPSIN (bundled
``resources/opsin/jre``, else ``JAVA_HOME`` / ``PATH``). Calls block, so the UI runs them in a
background thread. No Qt here.
"""
from __future__ import annotations

import atexit
import collections
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from functools import lru_cache

__all__ = ["CdkError", "CdkHelper", "cdk_jars", "cdk_available", "cdk_values", "shutdown",
           "CDK_METHODS", "CDK_LOGGING"]

# key -> (short label, description)
CDK_METHODS = {
    "jplogp": ("JPlogP", "JPlogP (Plante & Werner 2018), CDK"),
    "xlogp": ("XLogP", "XLogP v2 (Wang, Fu & Lai 1997), CDK"),
    "alogp": ("ALogP", "ALogP (Ghose, Viswanadhan & Wendoloski 1998), CDK"),
    "amr": ("AMR", "AMR molar refractivity (Ghose et al. 1998), CDK"),
}
# log4j's simple logger (on stderr): no log4j-core configuration, no extra Java modules
CDK_LOGGING = ("-Dlog4j2.loggerContextFactory="
               "org.apache.logging.log4j.simple.SimpleLoggerContextFactory")
START_TIMEOUT = 60.0


class CdkError(Exception):
    """The CDK helper is not available or failed."""


def cdk_jars():
    """The helper's class path - (helper jar, CDK module jars...) in the app resources, in the
    order of ``resources/java/cdk-classpath.txt`` - or None when anything is missing."""
    from rsstatchem.paths import resource
    helper, listing = resource("java/rsstatchem-cdk.jar"), resource("java/cdk-classpath.txt")
    if not helper.is_file() or not listing.is_file():
        return None
    try:
        names = [n.strip() for n in listing.read_text("utf-8").splitlines() if n.strip()]
    except OSError:
        return None
    jars = [resource("java/" + n) for n in names]
    if not jars or not all(j.is_file() for j in jars):
        return None
    return (helper, *jars)


def cdk_available(allow_system: bool = True) -> bool:
    from rsstatchem.core.chem_draw.names import find_java
    return cdk_jars() is not None and find_java(allow_system) is not None


class CdkHelper:
    """A persistent ``CdkProps`` Java process. Thread-safe; :meth:`compute` blocks."""

    def __init__(self, java: str | None = None, jars=None):
        self.java = java
        self.jars = jars
        self.version = ""
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr: collections.deque = collections.deque(maxlen=20)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.starts = 0

    # -- process management ------------------------------------------------------------------
    def _command(self) -> list[str]:
        from rsstatchem.core.chem_draw.names import find_java
        jars = self.jars or cdk_jars()
        java = self.java or find_java()
        if jars is None or java is None:
            raise CdkError("CDK is not available (it needs the CDK jars in resources/java, the "
                           "helper rsstatchem-cdk.jar and a Java runtime).")
        cp = os.pathsep.join(str(j) for j in jars)
        return [java, CDK_LOGGING, "-Dfile.encoding=UTF-8", "-Xss8m", "-Xshare:auto",
                "-XX:+UseSerialGC", "-cp", cp, "CdkProps"]

    def _start(self) -> None:
        cmd = self._command()
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1, **kw)
        except OSError as exc:
            raise CdkError(f"CDK helper could not be started ({exc}).") from None
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(proc.stdout, lines), daemon=True,
                         name="cdk-stdout").start()
        threading.Thread(target=self._drain, args=(proc.stderr,), daemon=True,
                         name="cdk-stderr").start()
        self._proc, self._lines = proc, lines
        self.starts += 1
        first = self._read(START_TIMEOUT)
        try:
            hello = json.loads(first)
        except (TypeError, ValueError):
            hello = {}
        if not hello.get("ready"):
            tail = " ".join(self._stderr)[-300:]
            self._kill()
            raise CdkError(f"CDK helper did not start: {tail or first or 'no answer'}")
        self.version = str(hello.get("cdk", ""))

    @staticmethod
    def _pump(stream, lines: queue.Queue) -> None:
        try:
            for line in stream:
                lines.put(line.rstrip("\n"))
        except (OSError, ValueError):
            pass
        lines.put(None)                     # end of stream: the process is gone

    def _drain(self, stream) -> None:
        try:
            for line in stream:
                self._stderr.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _read(self, timeout: float):
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                    proc.wait(timeout=3)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            self._kill()

    # -- requests ----------------------------------------------------------------------------
    def compute(self, smiles: str, timeout: float = 30.0) -> dict:
        """{'jplogp', 'xlogp', 'alogp', 'amr'} (None = not computable) for a SMILES. Raises
        CdkError when the helper cannot run; it is restarted on the next call."""
        smiles = " ".join(smiles.split())
        with self._lock:
            if not self.alive():
                self._kill()
                self._start()
            rid = str(next(self._ids))
            try:
                self._proc.stdin.write(f"{rid}\t{smiles}\n")
                self._proc.stdin.flush()
            except (OSError, ValueError, AttributeError):
                self._kill()
                raise CdkError("CDK helper stopped unexpectedly.") from None
            while True:
                line = self._read(timeout)
                if line is None:            # timeout or the process died: restart next time
                    died = not self.alive()
                    self._kill()
                    raise CdkError("CDK helper stopped unexpectedly." if died else
                                   "CDK helper did not answer in time.")
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue                # not an answer line
                if reply.get("id") == rid:
                    return {k: (float(reply[k]) if reply.get(k) is not None else None)
                            for k in CDK_METHODS} | {"error": reply.get("error", "")}


_HELPER: CdkHelper | None = None
_HELPER_LOCK = threading.Lock()


def _helper() -> CdkHelper:
    global _HELPER
    with _HELPER_LOCK:
        if _HELPER is None:
            _HELPER = CdkHelper()
        return _HELPER


def shutdown() -> None:
    """Stop the helper process (also done at interpreter exit)."""
    global _HELPER
    with _HELPER_LOCK:
        helper, _HELPER = _HELPER, None
    if helper is not None:
        helper.close()


@lru_cache(maxsize=512)
def cdk_values(smiles: str, timeout: float = 30.0) -> dict:
    """CDK logP / AMR values of one molecule (cached per SMILES). Raises CdkError."""
    return _helper().compute(smiles, timeout)


def _atexit() -> None:
    try:
        shutdown()
    except Exception:  # noqa: BLE001 - never fail at exit
        pass


atexit.register(_atexit)
