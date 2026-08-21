"""Colour vocabulary for the report.

Three separate jobs, three separate palettes -- mixing them is how figures end up
unreadable:

* **Clusters** are arbitrary integer labels, so they need a *qualitative* palette
  with no implied order.  HDBSCAN routinely returns 40-80 clusters, far more than
  any hand-picked palette covers, so we cycle a 12-hue colour-blind-safe base
  (Paul Tol's bright/light sets) and shift lightness and saturation on each pass.
  Two clusters sharing a hue are then separated by value, which survives both
  deuteranopia and a greyscale printer.
* **Features** (the cenSat / RepeatMasker vocabulary) are *not* arbitrary: the
  reader already has a mental colour map for satellite classes.  Alpha-satellite
  HOR is warm, HSat is cool, gamma/beta satellite green, rDNA magenta, and the
  transition/unannotated classes are deliberately grey so real signal pops.  This
  mapping is fixed so that two different runs are directly comparable.
* **Continuous** values use the matplotlib perceptually-uniform family: viridis
  for counts, cividis (which is explicitly optimised for CVD) for GC.

Everything here is pure-stdlib and deterministic: the same cluster id always gets
the same colour for a given ordering, so re-running the report is byte-stable.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence
from typing import Final

__all__ = [
    "BASE12",
    "NOISE_COLOR",
    "MISSING_COLOR",
    "FEATURE_COLORS",
    "SUPERPOP_COLORS",
    "VIRIDIS",
    "CIVIDIS",
    "DIVERGING_DARK",
    "DIVERGING_LIGHT",
    "qualitative_palette",
    "feature_color",
    "categorical_colors",
    "scale_stops",
]

# --------------------------------------------------------------------------
# qualitative
# --------------------------------------------------------------------------

#: Twelve hues drawn from Paul Tol's "bright" and "light" schemes.  Chosen so
#: that adjacent entries differ in hue *and* in value, because a legend is read
#: top-to-bottom and neighbours are what get confused.
BASE12: Final[tuple[str, ...]] = (
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#EE8866",  # orange
    "#44BB99",  # teal
    "#BBCC33",  # olive
    "#AA4499",  # magenta
    "#99DDFF",  # pale blue
    "#DDCC77",  # sand
)

#: (delta-lightness, delta-saturation) applied on successive passes over BASE12.
#: Positive lightness moves toward white, negative toward black.
_CYCLES: Final[tuple[tuple[float, float], ...]] = (
    (0.00, 0.00),
    (-0.34, 0.10),
    (0.36, -0.14),
    (-0.58, -0.22),
    (0.62, 0.22),
)

#: Noise / unassigned.  Mid grey reads as "not a cluster" in both themes.
NOISE_COLOR: Final[str] = "#606874"
#: Missing or NaN continuous values.
MISSING_COLOR: Final[str] = "#4a5058"


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb)


def _shift(color: str, dl: float, ds: float) -> str:
    """Lighten/darken ``color`` while keeping its hue."""
    red, green, blue = _hex_to_rgb(color)
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    light = light + dl * (1.0 - light) if dl > 0 else light * (1.0 + dl)
    light = min(0.90, max(0.14, light))
    sat = min(1.0, max(0.16, sat * (1.0 + ds)))
    return _rgb_to_hex(colorsys.hls_to_rgb(hue, light, sat))


def qualitative_palette(n: int) -> list[str]:
    """``n`` visually distinct colours, stable for any ``n``.

    Entry ``i`` never changes when ``n`` grows, so a cluster keeps its colour if
    the report is rebuilt after a run gains clusters.
    """
    if n <= 0:
        return []
    out: list[str] = []
    base = len(BASE12)
    for i in range(n):
        cycle = (i // base) % len(_CYCLES)
        # Beyond 5 passes we keep going, nudging lightness a little further each
        # time rather than repeating exactly.
        extra = i // (base * len(_CYCLES))
        dl, ds = _CYCLES[cycle]
        if extra:
            dl = dl + (0.12 * extra if dl >= 0 else -0.12 * extra)
            ds = ds - 0.08 * extra
        out.append(_shift(BASE12[i % base], dl, ds))
    return out


def categorical_colors(levels: Sequence[str], *, noise_labels: Sequence[str] = ()) -> list[str]:
    """Palette for arbitrary string levels; ``noise_labels`` always get grey."""
    noise = set(noise_labels)
    colors: list[str] = []
    palette = qualitative_palette(max(1, len(levels)))
    idx = 0
    for level in levels:
        if level in noise:
            colors.append(NOISE_COLOR)
        else:
            colors.append(palette[idx % len(palette)])
            idx += 1
    return colors


# --------------------------------------------------------------------------
# semantic feature palette
# --------------------------------------------------------------------------

#: Fixed colours for :data:`kmer_dust.schemas.FEATURE_VOCAB` plus the sentinel
#: values the annotation stage can emit.  Warm = alpha satellite, cool = HSat,
#: green = gamma/beta satellite, magenta = rDNA, grey = transition/unannotated,
#: desaturated slate = interspersed repeats (they are background, not signal).
FEATURE_COLORS: Final[dict[str, str]] = {
    # alpha satellite -- warm
    "asat_hor_active": "#E4572E",
    "asat_hor": "#F0913D",
    "asat_mon": "#F6C97A",
    "mon": "#D9C3A0",
    # HSat -- cool
    "hsat1a": "#4F86C6",
    "hsat1b": "#7FB3E0",
    "hsat2": "#2B5F8E",
    "hsat3": "#A6CBE8",
    # other satellites
    "bsat": "#2F8F5B",
    "gsat": "#71C078",
    "censat_other": "#8C7BA6",
    "subterminal": "#C08497",
    "rdna": "#D64FA0",
    "ct": "#7C838D",
    # RepeatMasker classes -- deliberately muted
    "line": "#6E7B8B",
    "sine": "#93A1B0",
    "ltr": "#57657A",
    "dna": "#7B6E8B",
    "satellite": "#B07AA1",
    "simple_repeat": "#9AA1AA",
    "low_complexity": "#B6BCC4",
    "rrna": "#B98BD1",
    "trna": "#A67FBF",
    "snrna": "#8E7AC0",
    "retroposon": "#6C7F94",
    "rc": "#7F7F7F",
    "repeat_unknown": "#6B7280",
    # extras
    "segdup": "#17A398",
    "telomere": "#6A4C93",
    "gene": "#5B7C99",
    # sentinels
    "unannotated": "#565D67",
    "": "#4A5058",
}


def feature_color(feature: str) -> str:
    """Colour for a feature name, falling back to a stable hashed hue."""
    known = FEATURE_COLORS.get(feature)
    if known:
        return known
    # Unknown vocabulary (a future track): derive a repeatable colour rather
    # than crashing or collapsing everything to one grey.
    index = sum(ord(ch) for ch in feature) % len(BASE12)
    return _shift(BASE12[index], -0.2, -0.35)


#: 1000 Genomes superpopulations.  These have a conventional colouring in the
#: literature; keeping close to it makes the plot legible without a legend.
SUPERPOP_COLORS: Final[dict[str, str]] = {
    "AFR": "#D9A441",
    "AMR": "#C0504D",
    "EAS": "#4E9A51",
    "EUR": "#4A7FB5",
    "SAS": "#8E6BAF",
    "OCE": "#3FA7A0",
    "": "#606874",
}


# --------------------------------------------------------------------------
# continuous
# --------------------------------------------------------------------------

VIRIDIS: Final[tuple[str, ...]] = (
    "#440154",
    "#482878",
    "#3E4989",
    "#31688E",
    "#26828E",
    "#1F9E89",
    "#35B779",
    "#6ECE58",
    "#B5DE2B",
    "#FDE725",
)

CIVIDIS: Final[tuple[str, ...]] = (
    "#00204D",
    "#00306F",
    "#39486B",
    "#575D6D",
    "#707173",
    "#8A8779",
    "#A69D75",
    "#C4B56C",
    "#E4CF5B",
    "#FFEA46",
)

#: Diverging scale for log2 enrichment.  The midpoint matches the card
#: background of the corresponding theme so "no enrichment" disappears instead
#: of shouting -- a white midpoint on a dark card is the single most common way
#: a diverging heatmap goes wrong.
DIVERGING_DARK: Final[tuple[str, ...]] = (
    "#5EA6E0",
    "#3C7FB8",
    "#2C4E68",
    "#191D23",
    "#6B3F22",
    "#C4692B",
    "#F0A04B",
)

DIVERGING_LIGHT: Final[tuple[str, ...]] = (
    "#2166AC",
    "#67A9CF",
    "#D1E5F0",
    "#F7F7F5",
    "#FDDBC7",
    "#EF8A62",
    "#B2182B",
)


def scale_stops(colors: Sequence[str]) -> list[list[object]]:
    """``[[position, color], ...]`` in the shape plotly expects."""
    n = len(colors)
    if n == 0:
        return []
    if n == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    return [[round(i / (n - 1), 6), colors[i]] for i in range(n)]
