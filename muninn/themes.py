"""Muninn Dark / Muninn Light themes.

Colors and roles are derived from the knotpm design system
(`~/code/knotpm/design/README.md`):

  - Plain surfaces, no gradients, 1px borders for elevation.
  - One brand color (teal) used sparingly; blue is functional (link/focus);
    orange is warn; red is danger.
  - "Pure black is reserved for ink only, never as a background."

The web design system has more semantic roles (text-muted, border, link,
brand-strong, danger-strong, ...) than Textual's color system exposes
(primary/secondary/accent/warning/error/success/foreground/background/
surface/panel). We collapse the ones that don't have a slot into Textual
`variables` only when it actually changes the rendered chrome - otherwise
they're left to derive from the base palette.
"""
from __future__ import annotations

from textual.theme import Theme

# --- Source palette (verbatim from knotpm design/README.md) ----------------
# Ink
INK_BLACK = "#000000"          # text only, never background
INK_242424 = "#242424"         # dark page canvas
INK_353535 = "#353535"         # dark surface (cards, code blocks)
INK_494949 = "#494949"         # dark surface-2 / border / light text-muted
INK_666666 = "#666666"         # dark text-subtle / light text-muted alt
INK_E2E2E2 = "#e2e2e2"         # dark text / light surface + border

# Teal (brand)
TEAL_LIGHT = "#6dcfc0"         # dark brand
TEAL = "#1aab90"               # dark brand-strong / light could-use
TEAL_DARK = "#0c7864"          # light brand
TEAL_DARKEST = "#004a3c"       # light brand-strong

# Blue (link / focus)
BLUE_LIGHT = "#27a3ff"         # dark link / focus
BLUE = "#003fe2"               # light link / focus
BLUE_DARK = "#0a138a"          # light link-strong (visited/hover)

# Signal (warn / danger)
ORANGE = "#f86a1d"             # warn (both themes)
RED = "#db0000"                # danger (both themes)
RED_DARK = "#99042e"           # danger-strong (both themes)

# Light page canvas (the only color outside the source list)
WHITE = "#ffffff"


MUNINN_DARK = Theme(
    name="muninn-dark",
    dark=True,
    # Brand teal carries identity. Light tint on dark surfaces, per the
    # design system's dark-theme mapping (brand: #6dcfc0).
    primary=TEAL_LIGHT,
    secondary=TEAL,
    # Functional blue is the accent (link/focus). Used by the Muninn|Huginn
    # divider border in app.py and by Textual's default focus chrome.
    accent=BLUE_LIGHT,
    warning=ORANGE,
    error=RED,
    # No green in the palette; "ok" status reads as brand-active so the OK
    # banner ends up teal, matching the brand-as-affirmation principle.
    success=TEAL,
    foreground=INK_E2E2E2,
    # Pure black would halate on OLED (per design doc); use ink #242424.
    background=INK_242424,
    surface=INK_353535,
    panel=INK_494949,
    variables={
        # Footer key glyph picks up the brand teal so the bottom chrome
        # echoes the top divider color.
        "footer-key-foreground": TEAL_LIGHT,
        # Link / link-strong don't have first-class Textual slots but Markdown
        # widgets pick these up.
        "link-color": BLUE_LIGHT,
        "link-color-hover": BLUE,
        "link-background-hover": INK_494949,
        # Border default = surface-2 from the design tokens.
        "border": INK_494949,
        # Pressed-danger maps to danger-strong from the design tokens.
        "error-darken-1": RED_DARK,
    },
)


MUNINN_LIGHT = Theme(
    name="muninn-light",
    dark=False,
    primary=TEAL_DARK,
    secondary=TEAL_DARKEST,
    accent=BLUE,
    warning=ORANGE,
    error=RED,
    success=TEAL_DARK,
    foreground=INK_BLACK,
    background=WHITE,
    # Light theme collapses surface and surface-2 to the same #e2e2e2 in the
    # design system; Textual treats them as distinct so we duplicate.
    surface=INK_E2E2E2,
    panel=INK_E2E2E2,
    variables={
        "footer-key-foreground": TEAL_DARK,
        "link-color": BLUE,
        "link-color-hover": BLUE_DARK,
        "link-background-hover": INK_E2E2E2,
        "border": INK_E2E2E2,
        "error-darken-1": RED_DARK,
    },
)


ALL_THEMES: tuple[Theme, ...] = (MUNINN_DARK, MUNINN_LIGHT)

# muninn-dark is the default if the user hasn't overridden via config or env.
DEFAULT_THEME_NAME: str = MUNINN_DARK.name
