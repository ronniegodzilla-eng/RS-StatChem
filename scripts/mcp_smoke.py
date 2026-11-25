"""Smoke test of the Claude connector as a client sees it: start a command that serves MCP over
stdio (the packaged app with ``--mcp``), list the tools and run an analysis.

    python scripts/mcp_smoke.py dist/RS-StatChem.app/Contents/MacOS/RS-StatChem
    python scripts/mcp_smoke.py dist/RS-StatChem/RS-StatChem.exe
    python scripts/mcp_smoke.py .venv/bin/python run.py          (development checkout)

Used by CI after PyInstaller (the Windows build is a windowed program, so this is the check
that its stdin/stdout really work as pipes). Both handshakes are tried: the 2026 protocol and
the older ``initialize`` one that current desktop clients use. Exits 1 on any failure.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "contoh_penelitian.rsd"
TIMEOUT = 300


async def check(command: list[str], mode: str, home: str) -> None:
    params = StdioServerParameters(command=command[0], args=[*command[1:], "--mcp"],
                                   env={"RSSTATCHEM_HOME": home})
    started = time.perf_counter()
    with anyio.fail_after(TIMEOUT):
        async with Client(params, mode=mode) as client:
            info = client.server_info
            tools = [t.name for t in (await client.list_tools()).tools]
            assert len(tools) >= 50 and "open_data" in tools, tools
            opened = await client.call_tool("open_data", {"path": str(DATA)})
            assert not opened.is_error, opened.content
            result = await client.call_tool("independent_samples_t_test", {
                "variables": ["posttest"], "group_variable": "kelompok"})
            text = result.content[0].text
            assert not result.is_error and "Independent Samples Test" in text, text
            menu = await client.call_tool("run_command", {
                "command": "Legacy Dialogs > 2 Independent Samples",
                "variables": {"vars": ["posttest"], "group": "kelompok"}, "press": "OK"})
            assert not menu.is_error and "Mann-Whitney U" in str(menu.content), menu.content
            drawn = await client.call_tool("chem_draw", {"structure": "CCO"})
            assert not drawn.is_error and "C2H6O" in drawn.content[0].text, drawn.content
            assert drawn.content[1].type == "image"
            print(f"mcp smoke ok ({mode}): {info.name} {info.version}, protocol "
                  f"{client.protocol_version}, {len(tools)} tools, "
                  f"{time.perf_counter() - started:.1f} s")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    command = [str(Path(argv[0]).absolute()) if Path(argv[0]).exists() else argv[0], *argv[1:]]
    with tempfile.TemporaryDirectory() as home:
        for mode in ("auto", "legacy"):
            try:
                anyio.run(check, command, mode, home)
            except BaseException as exc:  # noqa: BLE001 - report and fail the CI step
                import traceback

                traceback.print_exception(exc)
                print(f"mcp smoke FAILED ({mode})")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
