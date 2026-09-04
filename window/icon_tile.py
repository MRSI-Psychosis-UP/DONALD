"""Square icon tile that scales with the panel it lives in.

The right-panel analysis tiles have to satisfy two things at once: stay square,
and grow when the user drags the side panel wider. A fixed size gives the first
and not the second; a plain ``Expanding`` policy gives the second and not the
first, because the panel's trailing stretch absorbs spare height while the grid
still hands out spare width -- so the tiles flatten out as the panel widens.

``SquareIconTile`` gets both by driving height from width
(``heightForWidth``) and re-deriving the icon size on every resize, so one
number -- the available column width -- decides everything.
"""

from __future__ import annotations

try:
    from PyQt6.QtCore import QSize
    from PyQt6.QtWidgets import QPushButton, QSizePolicy
except Exception:  # PyQt5 fallback, matching the rest of the app
    from PyQt5.QtCore import QSize
    from PyQt5.QtWidgets import QPushButton, QSizePolicy


def _policy(horizontal, vertical):
    if hasattr(QSizePolicy, "Policy"):
        return QSizePolicy(getattr(QSizePolicy.Policy, horizontal), getattr(QSizePolicy.Policy, vertical))
    return QSizePolicy(getattr(QSizePolicy, horizontal), getattr(QSizePolicy, vertical))


class SquareIconTile(QPushButton):
    """Icon-only button that stays square and sizes its icon from its width."""

    #: Icon edge as a fraction of the tile edge. The tile carries 6px padding
    #: plus a border, so anything much above this starts to look cramped.
    ICON_RATIO = 0.68

    #: Floor, so the tiles stay usable when the panel is dragged narrow.
    MIN_SIDE = 56

    #: Ceiling, so a very wide panel does not produce absurd tiles.
    MAX_SIDE = 220

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        policy = _policy("Expanding", "Preferred")
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumSize(QSize(self.MIN_SIDE, self.MIN_SIDE))
        self.setMaximumWidth(self.MAX_SIDE)

    # heightForWidth is declared because it costs nothing and helps layouts
    # that honour it, but it is NOT what keeps the tile square: a QGridLayout
    # gives each row its hint height and lets the parent's trailing stretch
    # take the slack, so the tile would grow wide and stay short. The height
    # is therefore set explicitly from the width in resizeEvent.
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._side_for(width)

    def sizeHint(self) -> QSize:
        side = self._side_for(self.width() or self.MIN_SIDE)
        return QSize(side, side)

    def _side_for(self, width: int) -> int:
        return max(self.MIN_SIDE, min(int(width), self.MAX_SIDE))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        side = self._side_for(self.width())
        if self.height() != side:
            # Converges: this re-enters resizeEvent once, and the second pass
            # finds height already equal to side and stops.
            self.setFixedHeight(side)
        self._rescale_icon(side)

    def _rescale_icon(self, side: int = None) -> None:
        if side is None:
            side = self._side_for(self.width())
        edge = max(12, int(side * self.ICON_RATIO))
        if self.iconSize().width() != edge:
            self.setIconSize(QSize(edge, edge))

    def setIcon(self, icon) -> None:  # noqa: N802 - Qt naming
        super().setIcon(icon)
        # A theme change re-applies icons; keep the size derived from the tile
        # rather than from whatever the caller last set globally.
        self._rescale_icon()
