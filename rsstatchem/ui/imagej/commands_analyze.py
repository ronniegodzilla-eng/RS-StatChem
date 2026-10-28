"""Image Analysis commands: Analyze menu and Edit > Selection. **Owned by WP-C.**

``register(registry)`` adds this area's commands (labels exactly as in
``rsstatchem.core.imaging.menu_layout.labels_of_area("analyze")``); the ImageJ bar calls it
at start-up and enables the matching menu entries. Optional ``selftest()`` (no Qt widgets) is
run by ``RS-StatChem --selftest``. See docs/IMAGEJ_CONTRACTS.md.
"""
from __future__ import annotations


def register(registry) -> None:
    """Phase 0 stub: no commands yet."""
