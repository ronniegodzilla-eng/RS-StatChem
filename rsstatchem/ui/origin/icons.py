"""Inline-SVG icons for the Graph & Analysis workspace (own artwork, no image files).

The SVGs are registered in the Graph Studio icon table (:mod:`rsstatchem.ui.graph.gicons`)
and drawn through its cached :func:`~rsstatchem.ui.graph.gicons.icon`, whose cache the
app clears before the QApplication goes away.
"""
from __future__ import annotations

from rsstatchem.ui.graph import gicons

K = "#3a3a3a"     # outlines
B = "#1f5fbf"     # blue accents
R = "#d62728"     # red accents
G = "#2e8b3a"     # green accents
Y = "#e8a317"     # yellow / folder
_S = f'fill="none" stroke="{K}" stroke-width="1.5" stroke-linejoin="round"'
_AX = f'<path d="M3.5 3v17.5H21" fill="none" stroke="{K}" stroke-width="1.4"/>'


def _sheet(extra: str = "", head: str = B) -> str:
    return (f'<rect x="3" y="4" width="18" height="16" fill="#fff" stroke="{K}" '
            f'stroke-width="1.3"/><rect x="3" y="4" width="18" height="4" fill="{head}"/>'
            f'<path d="M9 8v12M15 8v12M3 12h18M3 16h18" stroke="#9a9a9a" stroke-width="0.8"/>'
            + extra)


def _badge(letter: str, color: str = B) -> str:
    return (f'<rect x="2.5" y="2.5" width="19" height="19" rx="2" fill="#fff" stroke="{K}" '
            f'stroke-width="1.2"/><text x="12" y="16.8" font-family="Arial" font-size="12" '
            f'font-weight="bold" text-anchor="middle" fill="{color}">{letter}</text>')


_SVG = {
    "o_new_project": (f'<path d="M5 3h9l5 5v13H5z" fill="#fff" stroke="{K}" stroke-width="1.4"/>'
                      f'<path d="M14 3v5h5" {_S}/><circle cx="16.5" cy="17.5" r="4.2" '
                      f'fill="{Y}"/><path d="M16.5 15v5M14 17.5h5" stroke="#fff" '
                      f'stroke-width="1.6"/>'),
    "o_new_book": _sheet('<circle cx="18" cy="18" r="4.3" fill="#2e8b3a"/><path d="M18 15.6v4.8'
                         'M15.6 18h4.8" stroke="#fff" stroke-width="1.6"/>'),
    "o_new_graph": (f'<rect x="3" y="3" width="18" height="18" fill="#fff" stroke="{K}" '
                    f'stroke-width="1.3"/><path d="M6 17l4-6 3 3 5-7" fill="none" '
                    f'stroke="{R}" stroke-width="1.8"/><circle cx="18" cy="18" r="4.3" '
                    f'fill="{G}"/><path d="M18 15.6v4.8M15.6 18h4.8" stroke="#fff" '
                    f'stroke-width="1.6"/>'),
    "o_new_notes": (f'<path d="M5 3h14v18H5z" fill="#fffbe0" stroke="{K}" stroke-width="1.3"/>'
                    f'<path d="M8 8h8M8 11.5h8M8 15h5" stroke="{K}" stroke-width="1.2"/>'),
    "o_new_folder": (f'<path d="M2.5 6.5h7l2 2h10v11h-19z" fill="{Y}" stroke="#a87400" '
                     f'stroke-width="1.2"/><path d="M15 12v6M12 15h6" stroke="#fff" '
                     f'stroke-width="1.8"/>'),
    "o_open": (f'<path d="M2.5 6.5h7l2 2h9v3" fill="{Y}" stroke="#a87400" stroke-width="1.2"/>'
               f'<path d="M2.5 19.5l3-8h16l-3 8z" fill="#f6c453" stroke="#a87400" '
               f'stroke-width="1.2"/><path d="M2.5 6.5v13" stroke="#a87400" stroke-width="1.2"/>'),
    "o_save": (f'<path d="M3.5 3.5h14l3 3v14h-17z" fill="{B}" stroke="#123e80" '
               f'stroke-width="1.2"/><rect x="7" y="3.5" width="9" height="6" fill="#fff"/>'
               f'<rect x="6" y="13" width="12" height="7.5" fill="#e9eef7"/>'),
    "o_import": (_sheet() + f'<path d="M12 1.5v8M8.7 6.4L12 9.7l3.3-3.3" fill="none" '
                 f'stroke="{G}" stroke-width="2.2"/>'),
    "o_import_excel": (f'<rect x="3" y="4" width="18" height="16" rx="1.5" fill="#1d6f42"/>'
                       f'<path d="M8 8.5l8 7M16 8.5l-8 7" stroke="#fff" stroke-width="2"/>'),
    "o_export": (f'<rect x="3" y="7" width="14" height="13" fill="#fff" stroke="{K}" '
                 f'stroke-width="1.3"/><path d="M5.5 17l3-4 2 2 3.5-5" fill="none" '
                 f'stroke="{R}" stroke-width="1.6"/><path d="M14 9.5L21 2.5M16 2.5h5v5" '
                 f'fill="none" stroke="{G}" stroke-width="2"/>'),
    "o_copy_page": (f'<rect x="8" y="8" width="13" height="13" fill="#fff" stroke="{K}" '
                    f'stroke-width="1.3"/><path d="M10 18l3-4 2 2 3.5-5" fill="none" '
                    f'stroke="{R}" stroke-width="1.5"/><path d="M16 8V3H3v13h5" {_S}/>'),
    "o_undo": (f'<path d="M9 14L4 9l5-5" fill="none" stroke="{B}" stroke-width="2.2" '
               f'stroke-linejoin="round"/><path d="M4 9h10a6 6 0 010 12h-3" fill="none" '
               f'stroke="{B}" stroke-width="2.2"/>'),
    "o_redo": (f'<path d="M15 14l5-5-5-5" fill="none" stroke="{B}" stroke-width="2.2" '
               f'stroke-linejoin="round"/><path d="M20 9H10a6 6 0 000 12h3" fill="none" '
               f'stroke="{B}" stroke-width="2.2"/>'),
    "o_refresh": (f'<path d="M19 12a7 7 0 11-2.1-5" fill="none" stroke="{G}" '
                  f'stroke-width="2.2"/><path d="M18.5 2.5v5h-5" fill="none" stroke="{G}" '
                  f'stroke-width="2.2"/>'),
    "o_duplicate": (f'<rect x="3" y="3" width="12" height="12" fill="#fff" stroke="{K}" '
                    f'stroke-width="1.3"/><rect x="9" y="9" width="12" height="12" fill="#fff" '
                    f'stroke="{B}" stroke-width="1.5"/>'),
    "o_explorer": (f'<rect x="2.5" y="3" width="19" height="18" fill="#fff" stroke="{K}" '
                   f'stroke-width="1.2"/><path d="M2.5 11.5h19" stroke="{K}"/>'
                   f'<path d="M5 5.5h4l1 1h4v3H5z" fill="{Y}"/><path d="M5 14h9M5 17h7" '
                   f'stroke="{B}" stroke-width="1.6"/>'),
    "o_results": (f'<rect x="3" y="3" width="18" height="18" fill="#fff" stroke="{K}" '
                  f'stroke-width="1.2"/><path d="M6 8h12M6 12h12M6 16h8" stroke="{G}" '
                  f'stroke-width="1.6"/>'),
    "o_messages": (f'<path d="M3 4h18v12H10l-5 4v-4H3z" fill="#fff" stroke="{K}" '
                   f'stroke-width="1.3"/><path d="M12 7v4.5" stroke="{B}" stroke-width="2"/>'
                   f'<circle cx="12" cy="13.6" r="1.1" fill="{B}"/>'),
    "o_objects": (f'<rect x="3" y="3" width="18" height="18" fill="#fff" stroke="{K}" '
                  f'stroke-width="1.2"/><path d="M6 7h4M8 7v10h4M8 12h4" fill="none" '
                  f'stroke="{K}" stroke-width="1.2"/><rect x="12.5" y="10.5" width="6" '
                  f'height="3" fill="{R}"/><rect x="12.5" y="15.5" width="6" height="3" '
                  f'fill="{B}"/>'),
    # column designations (Column toolbar)
    "o_set_x": _badge("X", R), "o_set_y": _badge("Y", B), "o_set_z": _badge("Z", G),
    "o_set_yerr": _badge("±", B), "o_set_label": _badge("L", "#7a3fbf"),
    "o_set_disregard": _badge("N", "#888"),
    "o_add_column": (_sheet() + f'<rect x="14" y="10" width="9" height="9" rx="1" fill="{G}"/>'
                     f'<path d="M18.5 11.5v6M15.5 14.5h6" stroke="#fff" stroke-width="1.7"/>'),
    "o_set_values": (f'<rect x="2.5" y="2.5" width="19" height="19" rx="2" fill="#fff" '
                     f'stroke="{K}" stroke-width="1.2"/><text x="12" y="16.5" '
                     f'font-family="Times New Roman" font-style="italic" font-size="12" '
                     f'text-anchor="middle" fill="{B}">f(x)</text>'),
    "o_sort": (f'<path d="M7 4v16M3.5 16.5L7 20l3.5-3.5" fill="none" stroke="{K}" '
               f'stroke-width="1.6"/><path d="M13 6h8M13 11h6M13 16h4" stroke="{B}" '
               f'stroke-width="2"/>'),
    "o_stats": (f'<rect x="3" y="3" width="18" height="18" fill="#fff" stroke="{K}" '
                f'stroke-width="1.2"/><text x="12" y="16.5" font-family="Times New Roman" '
                f'font-style="italic" font-size="13" text-anchor="middle" fill="{R}">x̄</text>'),
    "o_fit": (_AX + f'<circle cx="7" cy="16" r="1.4" fill="{K}"/><circle cx="11" cy="12.5" '
              f'r="1.4" fill="{K}"/><circle cx="15" cy="10.5" r="1.4" fill="{K}"/>'
              f'<circle cx="18.5" cy="6" r="1.4" fill="{K}"/><path d="M5 18L20.5 5" '
              f'stroke="{R}" stroke-width="1.8"/>'),
    # 2D Graphs toolbar
    "g_line": _AX + f'<path d="M5 17l4-7 4 4 7-9" fill="none" stroke="{B}" stroke-width="2"/>',
    "g_scatter": _AX + ''.join(f'<rect x="{x - 1.6}" y="{y - 1.6}" width="3.2" height="3.2" '
                               f'fill="{K}"/>' for x, y in ((7, 16), (10, 11), (14, 13),
                                                            (17, 7), (19.5, 9.5))),
    "g_line_symbol": _AX + f'<path d="M6 16l5-6 4 3 5-7" fill="none" stroke="{B}" ' \
                           f'stroke-width="1.5"/>' + ''.join(
        f'<rect x="{x - 1.7}" y="{y - 1.7}" width="3.4" height="3.4" fill="{R}"/>'
        for x, y in ((6, 16), (11, 10), (15, 13), (20, 6))),
    "g_yerror": _AX + ''.join(
        f'<path d="M{x} {y - 3.5}v7M{x - 1.5} {y - 3.5}h3M{x - 1.5} {y + 3.5}h3" '
        f'stroke="{K}" stroke-width="1"/><circle cx="{x}" cy="{y}" r="1.6" fill="{R}"/>'
        for x, y in ((7.5, 15), (12.5, 11), (17.5, 8))),
    "g_column": _AX + f'<rect x="6" y="11" width="3.4" height="9.5" fill="{B}"/>' \
                      f'<rect x="11" y="6" width="3.4" height="14.5" fill="{R}"/>' \
                      f'<rect x="16" y="13" width="3.4" height="7.5" fill="{G}"/>',
    "g_column_stacked": _AX + f'<rect x="7" y="12" width="4" height="8.5" fill="{B}"/>'
                              f'<rect x="7" y="7" width="4" height="5" fill="{R}"/>'
                              f'<rect x="14" y="14" width="4" height="6.5" fill="{B}"/>'
                              f'<rect x="14" y="9" width="4" height="5" fill="{R}"/>',
    "g_bar": _AX + f'<rect x="3.5" y="5" width="12" height="3.4" fill="{B}"/>'
                   f'<rect x="3.5" y="10" width="16" height="3.4" fill="{R}"/>'
                   f'<rect x="3.5" y="15" width="8" height="3.4" fill="{G}"/>',
    "g_area": _AX + f'<path d="M3.5 20.5V14l5-5 5 4 7.5-8v15.5z" fill="{B}" fill-opacity=".7" '
                    f'stroke="{B}" stroke-width="1.2"/>',
    "g_area_stacked": _AX + f'<path d="M3.5 20.5V15l5-3 5 2 7.5-4v10.5z" fill="{B}"/>'
                            f'<path d="M3.5 15V10l5-4 5 3 7.5-5v6l-7.5 4-5-2z" fill="{R}"/>',
    "g_pie": (f'<circle cx="12" cy="12" r="9" fill="{B}"/><path d="M12 12V3a9 9 0 017.8 13.5z" '
              f'fill="{R}"/><path d="M12 12l7.8 4.5A9 9 0 0112 21z" fill="{G}"/>'),
    "g_doughnut": (f'<circle cx="12" cy="12" r="7" fill="none" stroke="{B}" stroke-width="4"/>'
                   f'<path d="M12 5a7 7 0 016.1 10.5" fill="none" stroke="{R}" '
                   f'stroke-width="4"/>'),
    "g_histogram": _AX + ''.join(f'<rect x="{x}" y="{y}" width="3" height="{20.5 - y}" '
                                 f'fill="#7fa7e0" stroke="{K}" stroke-width=".8"/>'
                                 for x, y in ((5, 15), (8, 9), (11, 6), (14, 10), (17, 15))) +
                   f'<path d="M4.5 18C8 6 13 2 19.5 17" fill="none" stroke="{R}" '
                   f'stroke-width="1.4"/>',
    "g_box": _AX + f'<rect x="7" y="7" width="4.5" height="8" fill="#9ec2f0" stroke="{K}"/>'
                   f'<path d="M7 11h4.5M9.25 4v3M9.25 15v3.5M8 4h2.5M8 18.5h2.5" stroke="{K}"/>'
                   f'<rect x="14" y="9" width="4.5" height="6" fill="#f2a1a1" stroke="{K}"/>'
                   f'<path d="M14 12h4.5M16.25 6v3M16.25 15v3M15 6h2.5M15 18h2.5" stroke="{K}"/>',
    "g_double_y": (f'<path d="M3.5 3v17.5H20.5V3" fill="none" stroke="{K}" stroke-width="1.4"/>'
                   f'<path d="M5 16l4-6 4 3 6-8" fill="none" stroke="{K}" stroke-width="1.7"/>'
                   f'<path d="M5 8l5 5 4-2 5 6" fill="none" stroke="{R}" stroke-width="1.7"/>'),
    "g_stack": (f'<path d="M3.5 2.5v8H21M3.5 13v8H21" fill="none" stroke="{K}" '
                f'stroke-width="1.3"/><path d="M5 8l4-3 4 2 6-3" fill="none" stroke="{B}" '
                f'stroke-width="1.6"/><path d="M5 19l4-4 4 2 6-4" fill="none" stroke="{R}" '
                f'stroke-width="1.6"/>'),
    "g_waterfall": _AX + ''.join(f'<path d="M{5 + k} {18 - 3 * k}l3-4 3 3 4-6 2 3" fill="none" '
                                 f'stroke="{c}" stroke-width="1.5"/>'
                                 for k, c in ((0, B), (1.5, R), (3, G))),
    "g_bubble": _AX + f'<circle cx="8" cy="15" r="2.4" fill="{B}" fill-opacity=".75"/>'
                      f'<circle cx="14" cy="10" r="4" fill="{R}" fill-opacity=".7"/>'
                      f'<circle cx="18.5" cy="15.5" r="1.8" fill="{G}" fill-opacity=".8"/>',
    "g_color_scatter": _AX + ''.join(f'<circle cx="{x}" cy="{y}" r="1.9" fill="{c}"/>' for
                                     x, y, c in ((7, 16, "#3b4cc0"), (10, 12, "#6f92f3"),
                                                 (13, 13, "#c9d7f0"), (16, 9, "#f49a7b"),
                                                 (19, 6, "#b40426"))),
    "g_contour": (f'<rect x="3" y="3" width="18" height="18" fill="#3b4cc0"/>'
                  f'<ellipse cx="13" cy="11" rx="8" ry="6.5" fill="#6f92f3"/>'
                  f'<ellipse cx="13.5" cy="11" rx="5.5" ry="4.3" fill="#f7d36b"/>'
                  f'<ellipse cx="14" cy="11" rx="2.6" ry="2" fill="#d62728"/>'),
    "g_heatmap": ''.join(f'<rect x="{3 + 4.5 * (k % 4)}" y="{3 + 4.5 * (k // 4)}" width="4.5" '
                         f'height="4.5" fill="{c}"/>' for k, c in enumerate(
                             ["#3b4cc0", "#6f92f3", "#aac7fd", "#dddcdc", "#6f92f3", "#f7b89c",
                              "#f49a7b", "#aac7fd", "#aac7fd", "#f49a7b", "#d62728", "#f7b89c",
                              "#dddcdc", "#aac7fd", "#f7b89c", "#6f92f3"])),
    "g_3d": (f'<path d="M4 17l8 4 8-4M4 17V6M12 21v-6" fill="none" stroke="{K}" '
             f'stroke-width="1.2"/><path d="M4 12c3-6 6 2 8-1s5-6 8-2l-8 6z" fill="#f49a7b" '
             f'stroke="{R}" stroke-width="1"/>'),
    "g_polar": (f'<circle cx="12" cy="12" r="9" {_S}/><circle cx="12" cy="12" r="5" fill="none" '
                f'stroke="#999"/><path d="M12 3v18M3 12h18" stroke="#999"/><path d="M12 12l6-4 '
                f'1 6-5 4z" fill="none" stroke="{R}" stroke-width="1.6"/>'),
    "g_ternary": (f'<path d="M12 3L21 19H3z" {_S}/><circle cx="12" cy="12" r="1.6" fill="{R}"/>'
                  f'<circle cx="9" cy="15" r="1.6" fill="{B}"/><circle cx="15" cy="15.5" r="1.6"'
                  f' fill="{G}"/>'),
    # tools
    "t_reader": (f'<path d="M3 12h18M12 3v18" stroke="{K}" stroke-width="1.1"/>'
                 f'<circle cx="12" cy="12" r="4" fill="none" stroke="{R}" stroke-width="1.8"/>'
                 f'<path d="M15 15l5 5" stroke="{R}" stroke-width="2"/>'),
    # explorer items
    "e_folder": f'<path d="M2.5 6.5h7l2 2h10v11h-19z" fill="{Y}" stroke="#a87400"/>',
    "e_book": _sheet(),
    "e_report": _sheet(head="#7f7f7f") + f'<path d="M5 10h6" stroke="{R}" stroke-width="1.5"/>',
    "e_graph": (f'<rect x="3" y="3" width="18" height="18" fill="#fff" stroke="{K}" '
                f'stroke-width="1.3"/><path d="M6 17l4-6 3 3 5-7" fill="none" stroke="{R}" '
                f'stroke-width="1.8"/>'),
    "e_notes": (f'<path d="M5 3h14v18H5z" fill="#fffbe0" stroke="{K}" stroke-width="1.3"/>'
                f'<path d="M8 8h8M8 11.5h8M8 15h5" stroke="{K}" stroke-width="1.2"/>'),
    "e_column": (f'<rect x="8" y="3" width="8" height="18" fill="#fff" stroke="{K}"/>'
                 f'<rect x="8" y="3" width="8" height="4" fill="{B}"/>'),
}

for _name, _body in _SVG.items():
    gicons._SVG.setdefault(_name, _body)


def icon(name: str):
    """QIcon for ``name`` (Origin-workspace icons above, or a Graph Studio icon)."""
    return gicons.icon(name)
