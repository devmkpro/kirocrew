#!/usr/bin/env python3
"""Generate the fork's merged brand mark: Kiro Crew's ghost + Codebrain's brain.

ONE set of constants emits BOTH files, because the alternative -- hand-authoring
the SVG and separately hand-drawing the PNG -- guarantees they drift, and the two
are the same mark. The vector file is the artefact a designer edits; the raster is
what ``/logo.png`` actually serves.

Deliberately dependency-free apart from Pillow: this host has no SVG
rasteriser (no rsvg-convert, inkscape, ImageMagick or cairosvg), so the PNG is
DRAWN rather than rendered from the SVG. Every shape here is therefore expressed
as geometry both backends can reproduce exactly -- rounded rectangles, circles,
polygons and round-capped polylines -- and no bezier the two would interpolate
differently.

Run:  python packaging/logo/build_logo.py
Out:  src/kiro_crew/static/kirocrew-logo.svg
      src/kiro_crew/static/kirocrew-logo.png   (served at /logo.png)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
#: Drawn at 4x and downsampled: Pillow has no antialiased polygon fill, so
#: supersampling is what keeps the squircle and chevron edges clean.
SUPERSAMPLE = 4

# ── palette ────────────────────────────────────────────────────────────────────
PURPLE_LIGHT = (139, 92, 246)  # #8B5CF6 — top-left of the background sweep
PURPLE_DEEP = (109, 40, 217)  # #6D28D9 — bottom-right
BRAIN_OUTLINE = (76, 29, 149)  # #4C1D95 — Codebrain's shell, darkened to read on purple
BRAIN_FILL = (255, 255, 255)
CHEVRON = (79, 70, 229)  # #4F46E5 — Codebrain's own indigo
GHOST = (255, 255, 255)
EYE = (11, 11, 15)

# ── geometry, in 512-space ─────────────────────────────────────────────────────
BG_INSET = 6
BG_RADIUS = 118

#: Codebrain's brain: a rounded shell split down the middle, a chevron per half.
#: Centred on the canvas (x0 + x1 == SIZE) so the mark reads balanced at 16px.
BRAIN_BOX = (96, 130, 416, 380)
BRAIN_RADIUS = 94
BRAIN_STROKE = 30

CHEVRON_STROKE = 30
#: ``<`` in the left half, ``>`` in the right — the mark reads outward from the
#: divider, which is what makes it a *brain* and not a pair of brackets.
CHEVRON_LEFT = ((203, 218), (170, 255), (203, 292))
CHEVRON_RIGHT = ((309, 218), (342, 255), (309, 292))

#: Kiro Crew's ghost, top-left, clipped by the background corner exactly as the
#: original three-ghost mark clips its corner ghosts. Sized to sit BESIDE the
#: brain rather than over it: a ghost large enough to overlap the shell hides its
#: own eyes behind the brain's outline, which reads as a rendering fault.
GHOST_CENTER = (80, 62)
GHOST_RADIUS = 62
#: The tail giving the dome its ghost silhouette rather than a plain circle.
GHOST_TAIL = ((74, 58), (152, 92), (86, 120))
EYES = (((62, 74), 8, 12, -18.0), ((94, 82), 8, 12, -18.0))


def _svg() -> str:
    x0, y0, x1, y1 = BRAIN_BOX
    left = " ".join(f"{px},{py}" for px, py in CHEVRON_LEFT)
    right = " ".join(f"{px},{py}" for px, py in CHEVRON_RIGHT)
    tail = " ".join(f"{px},{py}" for px, py in GHOST_TAIL)
    eyes = "\n".join(
        f'      <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
        f'transform="rotate({angle} {cx} {cy})"/>'
        for (cx, cy), rx, ry, angle in EYES
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}"
     width="{SIZE}" height="{SIZE}" role="img" aria-label="Kiro Crew + Codebrain">
  <title>Kiro Crew + Codebrain</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="rgb{PURPLE_LIGHT}"/>
      <stop offset="1" stop-color="rgb{PURPLE_DEEP}"/>
    </linearGradient>
    <clipPath id="squircle">
      <rect x="{BG_INSET}" y="{BG_INSET}" width="{SIZE - 2 * BG_INSET}"
            height="{SIZE - 2 * BG_INSET}" rx="{BG_RADIUS}"/>
    </clipPath>
  </defs>

  <g clip-path="url(#squircle)">
    <rect x="{BG_INSET}" y="{BG_INSET}" width="{SIZE - 2 * BG_INSET}"
          height="{SIZE - 2 * BG_INSET}" rx="{BG_RADIUS}" fill="url(#bg)"/>
    <!-- Kiro Crew ghost: dome plus tail, clipped by the corner -->
    <g fill="rgb{GHOST}">
      <circle cx="{GHOST_CENTER[0]}" cy="{GHOST_CENTER[1]}" r="{GHOST_RADIUS}"/>
      <polygon points="{tail}"/>
    </g>
    <g fill="rgb{EYE}">
{eyes}
    </g>
  </g>

  <!-- Codebrain brain: shell, divider, one chevron per hemisphere -->
  <rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="{BRAIN_RADIUS}"
        fill="rgb{BRAIN_FILL}" stroke="rgb{BRAIN_OUTLINE}" stroke-width="{BRAIN_STROKE}"/>
  <line x1="{(x0 + x1) // 2}" y1="{y0}" x2="{(x0 + x1) // 2}" y2="{y1}"
        stroke="rgb{BRAIN_OUTLINE}" stroke-width="{BRAIN_STROKE}"/>
  <polyline points="{left}" fill="none" stroke="rgb{CHEVRON}"
            stroke-width="{CHEVRON_STROKE}" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="{right}" fill="none" stroke="rgb{CHEVRON}"
            stroke-width="{CHEVRON_STROKE}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def _gradient(size: int) -> Image.Image:
    """A diagonal two-stop sweep, matching the SVG's 0,0 -> 1,1 gradient.

    Built from a 2x2 image resized bicubically: the four corners carry the sweep's
    ends on the diagonal and their blend on the off-diagonal, which is what a
    linear gradient at 45 degrees is.
    """
    mid = tuple((a + b) // 2 for a, b in zip(PURPLE_LIGHT, PURPLE_DEEP))
    seed = Image.new("RGB", (2, 2))
    seed.putpixel((0, 0), PURPLE_LIGHT)
    seed.putpixel((1, 0), mid)
    seed.putpixel((0, 1), mid)
    seed.putpixel((1, 1), PURPLE_DEEP)
    return seed.resize((size, size), Image.BICUBIC)


def _rotated_ellipse(
    layer: Image.Image, center: tuple[int, int], rx: int, ry: int, angle: float, scale: int
) -> None:
    """Paste one rotated filled ellipse — Pillow cannot rotate a draw op."""
    cx, cy = (value * scale for value in center)
    rx, ry = rx * scale, ry * scale
    pad = max(rx, ry) * 2 + 4
    chip = Image.new("L", (pad, pad), 0)
    ImageDraw.Draw(chip).ellipse(
        (pad // 2 - rx, pad // 2 - ry, pad // 2 + rx, pad // 2 + ry), fill=255
    )
    chip = chip.rotate(angle, resample=Image.BICUBIC)
    colour = Image.new("RGBA", chip.size, (*EYE, 255))
    layer.paste(colour, (int(cx - pad // 2), int(cy - pad // 2)), chip)


def _png() -> Image.Image:
    scale = SUPERSAMPLE
    canvas = SIZE * scale

    def s(value: float) -> int:
        return int(round(value * scale))

    def pts(points) -> list[tuple[int, int]]:
        return [(s(px), s(py)) for px, py in points]

    # Background: the gradient shown only through the squircle, so the corners
    # stay genuinely transparent instead of being painted over.
    squircle = Image.new("L", (canvas, canvas), 0)
    ImageDraw.Draw(squircle).rounded_rectangle(
        (s(BG_INSET), s(BG_INSET), s(SIZE - BG_INSET), s(SIZE - BG_INSET)),
        radius=s(BG_RADIUS),
        fill=255,
    )
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    image.paste(_gradient(canvas).convert("RGBA"), (0, 0), squircle)

    # Ghost, drawn on its own layer and then masked by the squircle so it hugs
    # the corner exactly like the SVG's clip-path.
    ghost = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    gd = ImageDraw.Draw(ghost)
    gx, gy = GHOST_CENTER
    gd.ellipse(
        (s(gx - GHOST_RADIUS), s(gy - GHOST_RADIUS), s(gx + GHOST_RADIUS), s(gy + GHOST_RADIUS)),
        fill=(*GHOST, 255),
    )
    gd.polygon(pts(GHOST_TAIL), fill=(*GHOST, 255))
    for center, rx, ry, angle in EYES:
        _rotated_ellipse(ghost, center, rx, ry, angle, scale)
    image.paste(ghost, (0, 0), Image.composite(ghost.getchannel("A"), squircle.point(lambda _: 0), squircle))

    # Brain, on top and NOT clipped: it is the mark's centre.
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = BRAIN_BOX
    draw.rounded_rectangle(
        (s(x0), s(y0), s(x1), s(y1)),
        radius=s(BRAIN_RADIUS),
        fill=(*BRAIN_FILL, 255),
        outline=(*BRAIN_OUTLINE, 255),
        width=s(BRAIN_STROKE),
    )
    mid_x = s((x0 + x1) // 2)
    draw.line((mid_x, s(y0), mid_x, s(y1)), fill=(*BRAIN_OUTLINE, 255), width=s(BRAIN_STROKE))
    for chevron in (CHEVRON_LEFT, CHEVRON_RIGHT):
        draw.line(pts(chevron), fill=(*CHEVRON, 255), width=s(CHEVRON_STROKE), joint="curve")
        # Pillow's `joint` rounds the corner but leaves square ENDS; the SVG says
        # stroke-linecap="round", so the caps are drawn explicitly.
        radius = s(CHEVRON_STROKE) // 2
        for px, py in (chevron[0], chevron[-1]):
            draw.ellipse(
                (s(px) - radius, s(py) - radius, s(px) + radius, s(py) + radius),
                fill=(*CHEVRON, 255),
            )

    return image.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    static = Path(__file__).resolve().parents[2] / "src" / "kiro_crew" / "static"
    static.mkdir(parents=True, exist_ok=True)
    svg_path = static / "kirocrew-logo.svg"
    png_path = static / "kirocrew-logo.png"
    svg_path.write_text(_svg(), encoding="utf-8")
    _png().save(png_path, "PNG", optimize=True)
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
