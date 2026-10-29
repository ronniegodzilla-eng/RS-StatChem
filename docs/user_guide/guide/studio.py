"""Screenshots of the real app for the user guide, with numbered boxes and arrow callouts.

The app runs offscreen (``QT_QPA_PLATFORM=offscreen``) at twice the normal pixel density, so
every picture matches the current version and stays sharp in print. Annotations are placed from
the widgets' own geometry, so they follow the layout when it changes.

A scene function receives a :class:`Studio` and returns one :class:`Shot` (or a list of
them)::

    @scene("stats.data_view", "Data Editor dengan data contoh")
    def data_view(st):
        w = st.module("stats")
        shot = st.shot(w)
        shot.box(w.menuBar(), "Menu bar: semua perintah modul")
        shot.box(st.toolbar_button(w.toolbar, "Open"), "Membuka berkas data")
        shot.arrow(w.tabs.tabBar(), "Pindah Data View / Variable View", side="bottom")
        return shot

``box`` draws a red rounded rectangle with a numbered badge (explained in the legend under
the figure); ``arrow`` writes the explanation on the picture in a margin, with an arrow to the
target.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# a large virtual screen (long menus stay in one column) drawn at twice the pixel density; a fresh settings folder so no recent files or preferences leak into pictures
_TMP = Path(tempfile.mkdtemp(prefix="rsguide-"))
(_TMP / "screen.json").write_text(
    '{"synchronousWindowSystemEvents": true, "windowFrameMargins": false, "screens": '
    '[{"name": "guide", "x": 0, "y": 0, "width": 4800, "height": 3000, '
    '"logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1}]}', encoding="utf-8")
os.environ["QT_QPA_PLATFORM"] = f"offscreen:configfile={_TMP / 'screen.json'}"
os.environ["QT_SCALE_FACTOR"] = "2"
os.environ["RSSTATCHEM_NO_SPLASH"] = "1"
os.environ["RSSTATCHEM_HOME"] = str(_TMP / "home")
# keep matplotlib's font cache between builds (otherwise it is rebuilt in the fresh home)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsguide-matplotlib"))
# English number formatting in every picture, whatever the computer's region settings
os.environ["LANG"] = os.environ["LC_ALL"] = "en_US.UTF-8"

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import (QAction, QColor, QFont, QFontMetricsF, QImage, QPainter,  # noqa: E402
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QMenu, QMenuBar, QTabBar, QToolBar,  # noqa: E402
                               QWidget)

RED = QColor("#C62828")
RED_DARK = QColor("#8E1B1B")
SCALE = 2.0                 # device pixels per logical pixel in the saved PNGs
TITLE_H = 30                # synthetic window title bar (logical px)
SHADOW = 14                 # room for the drop shadow around a window
BADGE = 26                  # badge diameter


# ------------------------------------------------------------------------------ registry
@dataclass
class SceneInfo:
    key: str
    caption: str
    func: Callable


SCENES: dict[str, SceneInfo] = {}


def scene(key: str, caption: str = ""):
    """Register a screenshot scene under ``key`` (``module.name``)."""
    def deco(func):
        SCENES[key] = SceneInfo(key, caption, func)
        return func
    return deco


# ------------------------------------------------------------------------------ annotations
@dataclass
class Note:
    kind: str                 # "box" | "arrow" | "outline"
    rect: QRectF              # target, in shot coordinates (logical px, before chrome)
    text: str
    number: int | None = None
    side: str = "auto"


@dataclass
class Shot:
    """A captured window (or part of one) plus its annotations."""
    studio: "Studio"
    image: QImage
    origin: QWidget | None            # widget whose (0, 0) is the image's (0, 0)
    title: str | None = None          # window title for the synthetic title bar (None: none)
    notes: list[Note] = field(default_factory=list)
    overlays: list[tuple[QImage, QPointF]] = field(default_factory=list)
    crop_rect: QRectF | None = None

    # -- geometry ---------------------------------------------------------------------
    def rect_of(self, target, pad: float = 0) -> QRectF:
        """Shot coordinates of a widget, action, menu title, QRect or (x, y, w, h)."""
        if isinstance(target, QRectF):
            r = QRectF(target)
        elif isinstance(target, QRect):
            r = QRectF(target)
        elif isinstance(target, tuple):
            r = QRectF(*target)
        elif isinstance(target, QWidget):
            top_left = target.mapTo(self.origin, QPoint(0, 0)) if self.origin is not None \
                and target is not self.origin else QPoint(0, 0)
            r = QRectF(QPointF(top_left), QSize(target.width(), target.height()).toSizeF())
        else:
            raise TypeError(f"cannot locate {target!r}")
        return r.adjusted(-pad, -pad, pad, pad)

    # -- annotations ------------------------------------------------------------------
    def box(self, target, text: str, pad: float = 2) -> "Shot":
        """Red rounded rectangle with a numbered badge; ``text`` goes in the legend."""
        n = 1 + sum(1 for x in self.notes if x.number is not None)
        self.notes.append(Note("box", self.rect_of(target, pad), text, n))
        return self

    def arrow(self, target, text: str, side: str = "auto", pad: float = 2) -> "Shot":
        """A label written in the margin (``left``/``right``/``top``/``bottom``) with an
        arrow pointing at ``target``."""
        self.notes.append(Note("arrow", self.rect_of(target, pad), text, None, side))
        return self

    def outline(self, target, pad: float = 3) -> "Shot":
        """Red rectangle without a number (for instance the item chosen in a menu)."""
        self.notes.append(Note("outline", self.rect_of(target, pad), ""))
        return self

    def crop(self, target, pad: float = 6) -> "Shot":
        """Keep only this part of the window (and drop the title bar)."""
        self.crop_rect = self.rect_of(target, pad)
        self.title = None
        return self

    # -- menus ------------------------------------------------------------------------
    def menu(self, path: str, highlight: str | None = None) -> "Shot":
        """Show an opened menu, e.g. ``"Analyze > Compare Means"``; ``highlight`` outlines
        one item of the last menu shown. Menus are drawn where Qt would open them."""
        bar = self.origin.findChild(QMenuBar) if self.origin is not None else None
        if bar is None:
            raise ValueError("this window has no menu bar")
        parts = [p.strip() for p in path.split(">")]
        act = _find_action(bar.actions(), parts[0])
        pos = QPointF(self.rect_of(bar).topLeft()) + QPointF(bar.actionGeometry(act).bottomLeft())
        menu = act.menu()
        self.outline(QRectF(self.rect_of(bar).topLeft() +
                            QPointF(bar.actionGeometry(act).topLeft()),
                            QRectF(bar.actionGeometry(act)).size()), pad=0)
        for part in parts[1:] + [None]:
            img = self.studio.grab_menu(menu)
            self.overlays.append((img, QPointF(pos)))
            if part is None:
                break
            sub = _find_action(menu.actions(), part)
            geo = QRectF(menu.actionGeometry(sub))
            self.outline(QRectF(pos + geo.topLeft(), geo.size()), pad=0)
            pos = pos + QPointF(geo.right() - 2, geo.top() - 4)
            menu = sub.menu()
        if highlight:
            item = _find_action(menu.actions(), highlight)
            geo = QRectF(menu.actionGeometry(item))
            self.outline(QRectF(pos + geo.topLeft(), geo.size()), pad=0)
        return self

    def menu_item_rect(self, path: str) -> QRectF:
        """Shot rectangle of a menu item shown with :meth:`menu` (for ``box``/``arrow``)."""
        bar = self.origin.findChild(QMenuBar)
        parts = [p.strip() for p in path.split(">")]
        act = _find_action(bar.actions(), parts[0])
        pos = QPointF(self.rect_of(bar).topLeft()) + QPointF(bar.actionGeometry(act).bottomLeft())
        menu = act.menu()
        geo = QRectF(bar.actionGeometry(act))
        geo.moveTopLeft(self.rect_of(bar).topLeft() + geo.topLeft())
        for i, part in enumerate(parts[1:]):
            sub = _find_action(menu.actions(), part)
            g = QRectF(menu.actionGeometry(sub))
            geo = QRectF(pos + g.topLeft(), g.size())
            if i < len(parts) - 2:
                pos = pos + QPointF(g.right() - 2, g.top() - 4)
                menu = sub.menu()
        return geo

    def overlay(self, widget: QWidget, at, title: str | bool = False) -> "Shot":
        """Draw another window (a dialog, a popup) over this one; ``at`` is a point in shot
        coordinates or a target whose top-left corner is used. ``title`` (True: the window's
        own title) adds a title bar above the dialog, which then starts ``TITLE_H`` lower."""
        img = self.studio.grab(widget)
        if title:
            text = widget.windowTitle() if title is True else title
            img = _with_title_bar(img, text, self.studio.label_font)
        p = QPointF(at) if isinstance(at, (QPoint, QPointF)) else self.rect_of(at).topLeft()
        self.overlays.append((img, p))
        return self


def _with_title_bar(img: QImage, title: str, font: QFont) -> QImage:
    """The grabbed dialog with a synthetic title bar on top (same look as windows)."""
    w, h = img.width() / SCALE, img.height() / SCALE
    out = QImage(int(w * SCALE), int((h + TITLE_H) * SCALE),
                 QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(QColor("#FFFFFF"))
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.scale(SCALE, SCALE)
    _Renderer(None, font)._title_bar(p, QRectF(0, 0, w, TITLE_H), title)
    src = QImage(img)
    src.setDevicePixelRatio(1.0)
    p.drawImage(QRectF(0, TITLE_H, w, h), src)
    p.end()
    out.setDevicePixelRatio(SCALE)
    return out


def _find_action(actions, text: str) -> QAction:
    want = text.replace("&", "").rstrip(".").strip().lower()
    for a in actions:
        if a.isSeparator():
            continue
        if a.text().replace("&", "").split("\t")[0].rstrip(".").strip().lower() == want:
            return a
    names = [a.text().replace("&", "") for a in actions if not a.isSeparator()]
    raise KeyError(f"menu item {text!r} not found among {names}")


# ------------------------------------------------------------------------------ the studio
class Studio:
    """One offscreen app session for all scenes."""

    def __init__(self):
        from rsstatchem.app import AppContext
        from rsstatchem.ui.theme import apply_theme

        from PySide6.QtCore import QLocale

        QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.app = QApplication.instance() or QApplication([])
        apply_theme(self.app)
        self.ctx = AppContext(self.app)
        # pixel sizes: the same in measuring and in painting on the (scaled) image
        self.label_font = QFont(self.app.font())
        self.label_font.setPixelSize(15)
        self.label_font.setBold(True)

    # -- app helpers ------------------------------------------------------------------
    def settle(self, rounds: int = 6) -> None:
        for _ in range(rounds):
            self.app.processEvents()

    def module(self, key: str, path: str | None = None, size: tuple[int, int] | None = None):
        """A fresh main window of module ``key`` (optionally opening ``path``)."""
        from rsstatchem.ui import modules

        w = modules.open_module(self.ctx, key, path=path)
        if size:
            w.resize(*size)
        self.settle()
        return w

    def toolbar_button(self, toolbar: QToolBar, text: str) -> QWidget:
        """The button of a toolbar action, found by its text or tool tip."""
        want = text.replace("&", "").rstrip(".").lower()
        for act in toolbar.actions():
            names = (act.text(), act.toolTip(), act.iconText())
            if any(n and n.replace("&", "").rstrip(".").lower() == want for n in names):
                w = toolbar.widgetForAction(act)
                if w is not None:
                    return w
        raise KeyError(f"toolbar button {text!r} not found")

    def tab(self, bar: QTabBar, text: str) -> QRect:
        """Rectangle of a tab (in the tab bar's own coordinates) for ``Shot.box``."""
        for i in range(bar.count()):
            if bar.tabText(i).replace("&", "").lower() == text.lower():
                r = bar.tabRect(i)
                return QRect(bar.mapTo(bar.window(), r.topLeft()), r.size())
        raise KeyError(f"tab {text!r} not found")

    def close_all(self) -> None:
        for w in list(self.app.topLevelWidgets()):
            if w.isVisible():
                w.hide()
        self.settle()

    # -- grabbing ---------------------------------------------------------------------
    def grab(self, widget: QWidget) -> QImage:
        widget.show()
        self.settle()
        return widget.grab().toImage()

    def grab_menu(self, menu: QMenu) -> QImage:
        menu.ensurePolished()
        menu.adjustSize()
        menu.popup(QPoint(0, 0))
        self.settle()
        img = menu.grab().toImage()
        menu.hide()
        return img

    def shot(self, widget: QWidget, title: str | bool = True) -> Shot:
        """Capture a window; ``title`` True uses its window title for the title bar."""
        self.settle()
        img = self.grab(widget)
        if title is True:
            title = widget.windowTitle().replace("[*]", "") or "RS-StatChem"
        return Shot(self, img, widget, title or None)

    # -- rendering --------------------------------------------------------------------
    def render(self, shot: Shot, path: Path) -> list[tuple[int, str]]:
        """Write the annotated picture to ``path``; returns the legend [(number, text)]."""
        return _Renderer(shot, self.label_font).save(path)


# ------------------------------------------------------------------------------ rendering
class _Renderer:
    def __init__(self, shot: Shot, font: QFont):
        self.shot = shot
        self.font = font
        self.fm = QFontMetricsF(font)

    def save(self, path: Path) -> list[tuple[int, str]]:
        shot = self.shot
        base = QImage(shot.image)
        if abs(base.devicePixelRatio() - SCALE) > 1e-6:
            raise RuntimeError(f"screenshots must be grabbed at {SCALE}x, got "
                               f"{base.devicePixelRatio()}x")
        base.setDevicePixelRatio(1.0)
        w_log, h_log = base.width() / SCALE, base.height() / SCALE
        overlays = [(img, QRectF(at, QSize(int(img.width() / SCALE),
                                           int(img.height() / SCALE)).toSizeF()))
                    for img, at in shot.overlays]
        notes = [Note(n.kind, QRectF(n.rect), n.text, n.number, n.side) for n in shot.notes]
        if shot.crop_rect is not None:
            # a part of the window: flatten window + overlays, cut, no frame
            c = shot.crop_rect.intersected(QRectF(0, 0, w_log, h_log))
            flat = QImage(int(c.width() * SCALE), int(c.height() * SCALE),
                          QImage.Format.Format_ARGB32_Premultiplied)
            flat.fill(QColor("#FFFFFF"))
            fp = QPainter(flat)
            fp.scale(SCALE, SCALE)
            fp.translate(-c.topLeft())
            fp.drawImage(QRectF(0, 0, w_log, h_log), base)
            for img, r in overlays:
                fp.drawImage(r, img)
            fp.end()
            base, overlays = flat, []
            w_log, h_log = c.width(), c.height()
            for n in notes:
                n.rect.translate(-c.topLeft())

        frame_top = TITLE_H if shot.title else 0
        # everything in "shot" coordinates: window content at (0, 0), title bar above it
        win = QRectF(0, -frame_top, w_log, h_log + frame_top)
        extent = QRectF(win)
        for _, r in overlays:
            extent = extent.united(r)

        arrows = [n for n in notes if n.kind == "arrow"]
        for n in arrows:
            if n.side == "auto":
                n.side = "right" if n.rect.center().x() > w_log / 2 else "left"
        sides = {s: [n for n in arrows if n.side == s] for s in ("left", "right", "top", "bottom")}
        label_w = 230.0
        m_left = label_w + 40 if sides["left"] else 0
        m_right = label_w + 40 if sides["right"] else 0
        m_top = 54 if sides["top"] else 0
        m_bottom = 54 if sides["bottom"] else 0
        pad = SHADOW
        W = m_left + pad + extent.width() + pad + m_right
        H = m_top + pad + extent.height() + pad + m_bottom
        out = QImage(int(W * SCALE), int(H * SCALE), QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(QColor("#FFFFFF"))
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.scale(SCALE, SCALE)
        origin = QPointF(m_left + pad, m_top + pad) - extent.topLeft()

        def to_out(r: QRectF) -> QRectF:
            return r.translated(origin)

        win_rect = to_out(win)
        img_rect = to_out(QRectF(0, 0, w_log, h_log))
        self._shadow(p, win_rect, 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(win_rect)
        if shot.title:
            self._title_bar(p, QRectF(win_rect.left(), win_rect.top(), win_rect.width(),
                                      TITLE_H), shot.title)
        p.drawImage(img_rect, base)
        p.setPen(QPen(QColor("#B8B8B8"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(win_rect, 6 if shot.title else 2, 6 if shot.title else 2)
        for img, r in overlays:
            r = to_out(r)
            self._shadow(p, r, 6)
            p.drawImage(r, img)
            p.setPen(QPen(QColor("#BDBDBD"), 0.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)

        # outlines and boxes
        for n in notes:
            if n.kind in ("box", "outline"):
                r = to_out(n.rect)
                p.setPen(QPen(RED, 2.4 if n.kind == "box" else 2.0))
                p.setBrush(QColor(198, 40, 40, 18) if n.kind == "box" else Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r, 4, 4)
        # badges after all boxes, avoiding each other
        placed: list[QRectF] = []
        for n in notes:
            if n.kind == "box":
                self._badge(p, to_out(n.rect), n.number, placed,
                            QRectF(0, 0, W, H))
        # arrow labels
        for side, items in sides.items():
            if items:
                self._arrows(p, side, items, to_out, win_rect, W, H, label_w)
        p.end()
        out.setDevicePixelRatio(SCALE)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(path))
        return [(n.number, n.text) for n in notes if n.kind == "box"]

    # -- pieces -----------------------------------------------------------------------
    @staticmethod
    def _shadow(p: QPainter, r: QRectF, depth: int) -> None:
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(depth, 0, -1):
            alpha = int(28 * (1 - i / depth) ** 2) + 2
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(r.adjusted(-i, -i + 2, i, i + 2), 6 + i, 6 + i)
        p.restore()

    def _title_bar(self, p: QPainter, r: QRectF, title: str) -> None:
        p.save()
        path = QPainterPath()
        path.addRoundedRect(r.adjusted(0, 0, 0, 8), 6, 6)
        p.setClipRect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#EDEDED"))
        p.drawPath(path)
        p.setClipping(False)
        p.setPen(QPen(QColor("#D0D0D0"), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        for i, c in enumerate(("#C8C8C8", "#C8C8C8", "#C8C8C8")):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c))
            p.drawEllipse(QPointF(r.left() + 16 + i * 18, r.center().y()), 5.5, 5.5)
        f = QFont(self.font)
        f.setBold(False)
        f.setPixelSize(13)
        p.setFont(f)
        p.setPen(QColor("#4A4A4A"))
        text = QFontMetricsF(f).elidedText(title, Qt.TextElideMode.ElideMiddle, r.width() - 140)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def _badge(self, p: QPainter, r: QRectF, number: int, placed: list, canvas: QRectF):
        d = BADGE
        cands = [QPointF(r.left() - d * 0.45, r.top() - d * 0.45),
                 QPointF(r.right() - d * 0.55, r.top() - d * 0.45),
                 QPointF(r.left() - d * 0.45, r.bottom() - d * 0.55),
                 QPointF(r.right() - d * 0.55, r.bottom() - d * 0.55),
                 QPointF(r.left() + 4, r.top() + 4)]
        best = None
        for c in cands:
            br = QRectF(c, QSize(d, d).toSizeF())
            if not canvas.adjusted(2, 2, -2, -2).contains(br):
                continue
            if any(br.intersects(o.adjusted(-2, -2, 2, 2)) for o in placed):
                continue
            best = br
            break
        if best is None:
            best = QRectF(cands[0], QSize(d, d).toSizeF())
            while any(best.intersects(o) for o in placed):
                best.translate(d + 2, 0)
        placed.append(best)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(best.translated(0, 1.2))
        p.setPen(QPen(QColor("#FFFFFF"), 2))
        p.setBrush(RED)
        p.drawEllipse(best)
        f = QFont(self.font)
        f.setPixelSize(15)
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(best, Qt.AlignmentFlag.AlignCenter, str(number))
        p.restore()

    def _label_size(self, text: str, width: float) -> QSize:
        r = self.fm.boundingRect(QRectF(0, 0, width - 18, 400),
                                 int(Qt.TextFlag.TextWordWrap), text)
        return QSize(int(min(width, r.width() + 20)) + 1, int(r.height() + 11))

    def _arrows(self, p, side, items, to_out, win_rect, W, H, label_w):
        targets = [(n, to_out(n.rect)) for n in items]
        horizontal = side in ("left", "right")
        targets.sort(key=lambda t: t[1].center().y() if horizontal else t[1].center().x())
        boxes = []
        cursor = -1e9
        for n, tr in targets:
            size = self._label_size(n.text, label_w if horizontal else 170)
            if horizontal:
                y = max(tr.center().y() - size.height() / 2, cursor + 8, 6)
                x = 10 if side == "left" else W - size.width() - 10
                if side == "left":
                    x = win_rect.left() - SHADOW - 26 - size.width()
                else:
                    x = win_rect.right() + SHADOW + 26
                lr = QRectF(x, y, size.width(), size.height())
                cursor = lr.bottom()
            else:
                x = max(tr.center().x() - size.width() / 2, cursor + 10, 6)
                y = 8 if side == "top" else H - size.height() - 8
                lr = QRectF(x, y, size.width(), size.height())
                cursor = lr.right()
            boxes.append((n, tr, lr))
        for n, tr, lr in boxes:
            # arrow from the label edge to the nearest edge of the target
            if side == "left":
                a, b = QPointF(lr.right(), lr.center().y()), QPointF(tr.left(), _clamp(
                    lr.center().y(), tr.top() + 3, tr.bottom() - 3))
            elif side == "right":
                a, b = QPointF(lr.left(), lr.center().y()), QPointF(tr.right(), _clamp(
                    lr.center().y(), tr.top() + 3, tr.bottom() - 3))
            elif side == "top":
                a, b = QPointF(lr.center().x(), lr.bottom()), QPointF(_clamp(
                    lr.center().x(), tr.left() + 3, tr.right() - 3), tr.top())
            else:
                a, b = QPointF(lr.center().x(), lr.top()), QPointF(_clamp(
                    lr.center().x(), tr.left() + 3, tr.right() - 3), tr.bottom())
            p.save()
            p.setPen(QPen(RED, 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(tr, 4, 4)
            _arrow_line(p, a, b)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(RED)
            p.drawRoundedRect(lr, 6, 6)
            p.setFont(self.font)
            p.setPen(QColor("#FFFFFF"))
            p.drawText(lr.adjusted(9, 5, -9, -5),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft |
                           Qt.TextFlag.TextWordWrap), n.text)
            p.restore()


def _clamp(v, lo, hi):
    return max(lo, min(hi, v)) if lo <= hi else (lo + hi) / 2


def _arrow_line(p: QPainter, a: QPointF, b: QPointF) -> None:
    import math

    p.setPen(QPen(RED, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    ang = math.atan2(b.y() - a.y(), b.x() - a.x())
    head = 10
    tip = b
    base = QPointF(b.x() - head * math.cos(ang), b.y() - head * math.sin(ang))
    p.drawLine(a, base)
    left = QPointF(base.x() + head * 0.5 * math.sin(ang), base.y() - head * 0.5 * math.cos(ang))
    right = QPointF(base.x() - head * 0.5 * math.sin(ang), base.y() + head * 0.5 * math.cos(ang))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(RED)
    p.drawPolygon(QPolygonF([tip, left, right]))
