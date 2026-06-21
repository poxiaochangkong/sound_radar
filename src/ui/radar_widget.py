"""
Radar widget: paints a top-down radar with contacts.

Coordinate convention:
  - angle_deg = 0  -> front (up on screen)
  - angle_deg = 90 -> right
  - angle_deg = 180/-180 -> back (down on screen)
  - angle_deg = -90 -> left

The painter is resolution-independent: it queries its own size and scales.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (QBrush, QColor, QPainter, QPen, QRadialGradient,
                         QLinearGradient)
from PyQt6.QtWidgets import QWidget

from src.models import RadarContact


def _color(hex_str: str, alpha: int = 255) -> QColor:
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


class RadarWidget(QWidget):
    """Top-down radar display. Call set_contacts() to update."""

    def __init__(
        self,
        colors: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Minimum size so the widget is usable inside layouts.
        self.setMinimumSize(320, 320)

        c = colors or {}
        self._bg_color        = c.get("background",        "#0a0e14")
        self._grid_color      = c.get("grid_color",        "#1f2937")
        self._sweep_color     = c.get("sweep_color",       "#22d3ee")
        self._contact_color   = c.get("contact_color",     "#f59e0b")
        self._front_zone_color = c.get("front_zone_color", "#10b981")
        self._side_zone_color  = c.get("side_zone_color",  "#3b82f6")
        self._back_zone_color  = c.get("back_zone_color",  "#ef4444")

        self._contacts: List[RadarContact] = []
        # Sweep animation phase in radians.
        self._sweep_phase: float = 0.0
        # Enable periodic repaint so the sweep and decay animate even when
        # no new contacts arrive.
        self._tick_count: int = 0

    # ---- public API --------------------------------------------------------

    def set_contacts(self, contacts: List[RadarContact]) -> None:
        """Replace current contacts. Called by the UI thread (~60 Hz)."""
        self._contacts = list(contacts)
        self.update()

    def advance_sweep(self, dt_seconds: float) -> None:
        """Advance the radar sweep animation. Call from a timer."""
        # One full revolution every 3 seconds.
        self._sweep_phase = (self._sweep_phase + dt_seconds * (2 * 3.14159265 / 3.0)) % (2 * 3.14159265)
        self.update()

    # ---- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        radius = min(w, h) / 2.0 - 8.0
        if radius <= 0:
            return

        # Background.
        p.fillRect(self.rect(), _color(self._bg_color))

        # Faint zone wedges (front/side/back) to give orientation.
        self._draw_zones(p, cx, cy, radius)

        # Range rings + crosshairs.
        pen = QPen(_color(self._grid_color, 200), 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for frac in (0.33, 0.66, 1.0):
            p.drawEllipse(QPointF(cx, cy), radius * frac, radius * frac)

        # Vertical and horizontal crosshair.
        p.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))
        p.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))

        # Labels for F / B / L / R.
        p.setPen(_color(self._grid_color, 230))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        p.drawText(QRectF(cx - 12, 2, 24, 16), Qt.AlignmentFlag.AlignCenter, "F")
        p.drawText(QRectF(cx - 12, h - 18, 24, 16), Qt.AlignmentFlag.AlignCenter, "B")
        p.drawText(QRectF(2, cy - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, "L")
        p.drawText(QRectF(w - 18, cy - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, "R")

        # Sweep line.
        self._draw_sweep(p, cx, cy, radius)

        # Contacts.
        self._draw_contacts(p, cx, cy, radius)

        p.end()

    def _draw_zones(self, p: QPainter, cx: float, cy: float, radius: float) -> None:
        # Front zone: -45..45 deg. Side zones: -135..-45 and 45..135. Back: rest.
        # Screen angles: 0 deg = up. Convert bearing (0=front/up) directly.
        def _wedge(start_deg: float, span_deg: float, color_hex: str) -> None:
            # Qt angles are in 1/16 degree, measured counter-clockwise from 3 o'clock.
            # Our bearing 0 = up (12 o'clock), + clockwise. Map:
            # qt_start = (90 - bearing) * 16; qt_span = -span * 16.
            qt_start = int((90 - start_deg) * 16)
            qt_span = int(-span_deg * 16)
            grad = QRadialGradient(QPointF(cx, cy), radius)
            c = _color(color_hex, 38)
            grad.setColorAt(0.0, c)
            transparent = QColor(color_hex)
            transparent.setAlpha(0)
            grad.setColorAt(1.0, transparent)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPie(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius),
                      qt_start, qt_span)

        _wedge(-45, 90, self._front_zone_color)   # -45..+45
        _wedge(45, 90, self._side_zone_color)     # +45..+135
        _wedge(-135, 90, self._side_zone_color)   # -135..-45
        _wedge(135, 90, self._back_zone_color)    # +135..-135 (back)

    def _draw_sweep(self, p: QPainter, cx: float, cy: float, radius: float) -> None:
        # Sweep is a thin line + faint trailing wedge.
        # Screen angle: 0 = up. Our internal phase increases clockwise.
        phase = self._sweep_phase
        # Tip point.
        tip_x = cx + radius * float(__import__("math").sin(phase))
        tip_y = cy - radius * float(__import__("math").cos(phase))

        pen = QPen(_color(self._sweep_color, 220), 2)
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy), QPointF(tip_x, tip_y))

    def _draw_contacts(self, p: QPainter, cx: float, cy: float, radius: float) -> None:
        import math
        for c in self._contacts:
            # Map bearing to screen angle: 0 -> up, + -> clockwise.
            theta = math.radians(c.angle_deg)
            x = cx + radius * 0.9 * math.sin(theta)
            y = cy - radius * 0.9 * math.cos(theta)

            # Intensity drives alpha and dot radius.
            alpha = int(80 + 175 * max(0.0, min(1.0, c.intensity)))
            dot_r = 4.0 + 8.0 * c.intensity

            # Outer glow.
            glow = _color(self._contact_color, max(20, alpha // 3))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), dot_r * 2.2, dot_r * 2.2)

            # Inner dot.
            core = _color(self._contact_color, alpha)
            p.setBrush(QBrush(core))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), dot_r, dot_r)