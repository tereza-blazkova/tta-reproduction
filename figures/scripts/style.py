"""How this paper's figures look — one place.

One font stack, one type scale and fixed semantic roles. **The semantic colours are the point**: teal is
"tutor move" and peach "student signal" in every figure, so no figure needs a legend for them, and
the two providers are blue (TSL) and olive (Eedi).

The literals below are fixed, not derived.

Two rules kept from the original, both load-bearing:
  * a figure script chooses WHAT to draw; every size, colour and output rule lives here;
  * a silent font fallback is a hard error, not a warning. matplotlib substitutes DejaVu Sans
    and emits a UserWarning nobody reads, and the figure then ships in the wrong typeface.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "figures" / "input"
OUTPUT = ROOT / "figures" / "output"

#: Global size multiplier: PLOT_SCALE=1.8 renders a deck-sized copy with identical proportions
#: rather than a differently-designed figure. Design at the final size.
SCALE = float(os.environ.get("PLOT_SCALE", "1.0"))
_FONT_SCALE = 1.0
_PAPER = False                          # sizes from PAPER (main-text figures), see apply_style
_MARK = 1.0                             # factor on the shared panel functions' marks, see mark()

FONT_STACK = ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans"]

#: Text width of the manuscript: \the\textwidth of acmart `manuscript` is 430.0 pt = 5.95 in
#: (measured). The main-text figures are drawn at exactly this width, saved at exactly this width
#: (save(..., width=TEXT_W)) and included at \linewidth, so they print 1:1. Height is per figure.
TEXT_W = 5.95
#: The main-text figures' sizes in points, as printed (they print 1:1): one type scale for all
#: three. Axis labels (x and y), role titles and column headers 7 pt; tick labels, legends, colour
#: keys and code names 6 pt; notes 5.5 pt; numbers inside heat-map cells 5 pt; panel letters 9 pt.
PAPER = dict(font=7.0, title=7.0, label=7.0, tick=6.0, legend=6.0, code=6.0, note=5.5, cell=5.0,
             panel_letter=9.0, axes_lw=0.5, tick_len=2.0, tick_w=0.5)
STYLE = dict(
    fig_w=6.5, fig_h=5.6,
    # text sizes chosen for legibility in print
    font=8.0, title=8.7, label=8.0, tick=7.1, legend=7.1,
    panel_letter=10.4, note=6.7,
    code=7.0,            # code names in a list of codes (row labels): one size in every figure
    cell=6.0,            # numbers inside heat-map cells
    axes_lw=0.6, tick_len=2.4, tick_w=0.6, grid_lw=0.5,
    scatter_s=1.53, scatter_alpha=0.72, bar_edge_lw=0.5,
    dpi_raster=600, pad_inches=0.02,
)

INK = "#111111"        # titles, panel letters
INK_2 = "#4A4A4A"      # secondary marks (legend keys); text and axes are INK
INK_MUTED = "#8A8A8A"  # reference lines (zero, diagonal), never axes
GRID = "#E8E8E4"

#: The construct a column measures. Fill for marks, ink for text (code names, headers): the inks
#: are dark enough for small text (contrast on white 9.8 and 8.8) and as dark as each other.
#: Dark muted teal against light saturated peach (L 0.44 vs 0.80), so the pair holds in greyscale; every code name
#: also carries its role in its text colour, so the pair never carries identity alone.
TUTOR = "#1F5F5B"       # deep teal (L 0.44, C 0.06: dark, muted)
STUDENT = "#F6A97A"     # peach (L 0.80, C 0.11: light, saturated)
TUTOR_INK = "#123F3D"
STUDENT_INK = "#9A4E22"

#: One diverging scale for every "less / more than expected" grid (dusty purple below, muted ochre
#: above), so the timing heat map, Fig. 1's position column and the by-state table read the same way.
#: Purple (OKLCH hue 314) and ochre (78) sit away from teal (189) and peach (52), so the scale never
#: reads as a role. The ochre is lighter than the purple (L 0.74 vs 0.54).
#: The midpoint is a light neutral grey, not white, so "as expected" is a colour, not the background.
DIVERGING = ["#80618F", "#EDECE8", "#D5A044"]
DIVERGING_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", DIVERGING)

#: Providers. TSL is a dark blue; Eedi is olive, which stays clear of the tutor teal.
TSL = "#156FA8"
EEDI = "#6B7A2E"
NEUTRAL = "#636363"

#: The student's record (attainment measured on other sessions; stages 39 and 42), kept apart from
#: the code colours so "who the student is" never reads as a tutor move or a student signal.
ATTAINMENT = "#7A6A58"
#: The outcome of an answer, where points are coloured by it (fig_head_to_head).
OUTCOME_RIGHT = "#4F6D8F"
OUTCOME_WRONG = "#C8553D"


def tsize(key: str) -> float:
    if _PAPER:
        return PAPER[key] * SCALE
    return STYLE[key] * SCALE * _FONT_SCALE


def mark(x: float) -> float:
    """A line width or marker size (points) of the shared panel functions, which are sized for
    the supplementary figures' wider canvases; a main-text figure that draws them at the text
    width sets the factor in apply_style(mark=...)."""
    return x * _MARK


#: Files of other Helvetica designs that register under the family name "Helvetica" itself
#: (seen with the Compressed and Rounded Bold files of a user font install): they tie with the
#: real regular and bold faces, and matplotlib then picks one of them for body text.
_VARIANT_FILES = ("compressed", "condensed", "rounded", "narrow", "light", "thin", "black")


def _drop_variant_faces() -> None:
    keep = [f for f in fm.fontManager.ttflist
            if not (f.name in FONT_STACK and any(v in Path(f.fname).name.lower() for v in _VARIANT_FILES))]
    if len(keep) != len(fm.fontManager.ttflist):
        fm.fontManager.ttflist = keep
        cached = getattr(fm.fontManager, "_findfont_cached", None)
        if cached is not None:
            cached.cache_clear()


def apply_style(font_scale: float = 1.0, paper: bool = False, mark: float = 1.0) -> None:
    """`paper`: a main-text figure, drawn at TEXT_W and printed 1:1, with its sizes from PAPER
    (`font_scale` is then ignored). `mark`: factor on the shared panel functions' marks."""
    global _FONT_SCALE, _PAPER, _MARK
    _FONT_SCALE = 1.0 if paper else font_scale
    _PAPER, _MARK = paper, mark
    _drop_variant_faces()
    fam = next((f for f in FONT_STACK if _has(f)), None)
    if fam is None:
        raise SystemExit("FONT: none of " + ", ".join(FONT_STACK) + " is installed.")
    s = SCALE * _FONT_SCALE
    size = {k: tsize(k) for k in ("font", "title", "label", "tick", "legend")}
    line = ({k: PAPER[k] * SCALE for k in ("axes_lw", "tick_len", "tick_w")} if paper
            else {k: STYLE[k] * s for k in ("axes_lw", "tick_len", "tick_w")})
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam] + [f for f in FONT_STACK if f != fam] + ["DejaVu Sans"],
        "font.size": size["font"],
        # maths text ($^\\dagger$ on the residual codes) in the same face, not DejaVu Sans
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam, "mathtext.default": "regular",
        "axes.titlesize": size["title"], "axes.titlecolor": INK,
        "axes.labelsize": size["label"], "axes.labelcolor": INK,
        "axes.edgecolor": INK, "axes.linewidth": line["axes_lw"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.labelsize": size["tick"], "ytick.labelsize": size["tick"],
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.size": line["tick_len"], "ytick.major.size": line["tick_len"],
        "xtick.major.width": line["tick_w"], "ytick.major.width": line["tick_w"],
        "xtick.direction": "in", "ytick.direction": "in",          # ticks inside the plots
        "legend.fontsize": size["legend"], "legend.frameon": False,
        "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "lines.solid_capstyle": "round",
        # 42 = TrueType, so the PDF carries selectable, editable text rather than outlines.
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": STYLE["pad_inches"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    # A missing GLYPH is the same class of failure as a missing FAMILY: matplotlib draws a
    # box and warns, and the figure ships with a tofu where an arrow was meant to be. Helvetica
    # has no U+2190/U+2192, which this caught twice. Promote it to an error so the figure
    # cannot be rendered wrong quietly; say the direction in words instead.
    import warnings
    warnings.filterwarnings("error", message=r"Glyph \d+ .* missing from font",
                            category=UserWarning)
    resolved = Path(fm.findfont(fm.FontProperties(
        family=plt.rcParams["font.sans-serif"]))).name
    if resolved.startswith("DejaVu"):
        raise SystemExit(f"FONT FELL BACK TO {resolved} — refusing to render. Clear "
                         f"~/.cache/matplotlib and retry.")
    if any(v in resolved.lower() for v in _VARIANT_FILES):
        raise SystemExit(f"FONT RESOLVED TO A VARIANT DESIGN ({resolved}) — refusing to render.")
    print(f"[style] font -> {resolved} (scale {SCALE:g})", flush=True)


def _has(family: str) -> bool:
    try:
        fm.findfont(family, fallback_to_default=False)
        return True
    except ValueError:
        return False


#: the share of white in the light fills of bars and marks (Fig. 1's bars and marks, Fig. 2's
#: shares bars); their edges stay the role's dark shade
LIGHT = 0.35


def tint(colour, amount: float) -> tuple:
    """`colour` mixed with white; `amount` is the share of white. Markers are filled with a light
    tint of their colour and edged in its dark shade (Figures 1 and 2)."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


#: a very light grey behind every other row of a list of codes, so a row can be followed from its
#: label across the plot
ZEBRA = "#F3F3F0"


def zebra(ax, ys_top_to_bottom, x0: float = 0.0, x1: float = 1.0, half: float = 0.5) -> None:
    """Band every second row (the first row from the top stays white), inside the plot. `ys_top_to_bottom`:
    the rows' y positions in data units, in reading order; `x0`, `x1`: the band's extent in axes
    fraction (default the whole plot), drawn below everything else in the axes."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for k, y in enumerate(ys_top_to_bottom):
        if k % 2 == 1:
            ax.add_patch(Rectangle((x0, y - half), x1 - x0, 2 * half, transform=tr, facecolor=ZEBRA,
                                   lw=0, zorder=0, clip_on=True))


def panel_letter(ax, letter: str, dx: float = -0.085, dy: float = 1.045) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=tsize("panel_letter"),
            color=INK, fontweight="bold", va="bottom", ha="left")


def letter_top_left(fig, ax, letter: str, pad_pt: float = 3.0) -> None:
    """A panel letter at the panel's top left, its left edge where the panel's leftmost text starts
    (the y-axis label or the row names).

    Where it fits, the top of the letter is level with the top of the y axis. Where that would
    touch the y-axis label or a tick label, the letter sits just above the axes instead. One rule
    for every SI figure, so the letters sit in the same place throughout."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x0 = max(fig.transFigure.inverted().transform((ax.get_tightbbox(r).x0, 0))[0], 0.0)
    top = ax.get_position().y1
    t = fig.text(x0, top, letter, fontsize=tsize("panel_letter"), fontweight="bold",
                 color=INK, ha="left", va="top")
    bb = t.get_window_extent(r).expanded(1.0, 1.0)
    pad = 1.5 * fig.dpi / 72                          # keep 1.5 pt clear of any label
    bb = bb.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    lo, hi = sorted(ax.get_ylim())                    # tick labels outside the limits are not drawn
    others = [ax.yaxis.label] + [tl for tl, loc in zip(ax.get_yticklabels(), ax.get_yticks())
                                 if tl.get_visible() and tl.get_text() and lo <= loc <= hi]
    if any(o.get_window_extent(r).overlaps(bb) for o in others if o.get_text()):
        # next choice: still level with the top of the y axis, but left of the y-axis label,
        # as long as that stays inside the figure and clear of the other panels
        lab = ax.yaxis.label.get_window_extent(r)
        x_px = lab.x0 - 2.5 * fig.dpi / 72 - t.get_window_extent(r).width   # 2.5 pt from the label
        t.set_x(fig.transFigure.inverted().transform((x_px, 0))[0])
        nb = t.get_window_extent(r)
        clear = x_px >= 0 and not any(
            a is not ax and a.get_tightbbox(r).overlaps(nb) for a in fig.axes)
        if not clear:                                 # last choice: just above the axes
            t.set_x(x0)
            t.set_va("bottom")
            t.set_y(top + pad_pt / 72 / fig.get_figheight())


def save(fig, name: str, width: float | None = None) -> None:
    """Write `name`.pdf and .png. With `width` (a main-text figure: TEXT_W), the file is exactly
    that wide, the figure's own width, so it prints 1:1 at the text width; only the height is trimmed
    to the content. Anything drawn outside the figure's width is an error, not a wider file."""
    from matplotlib.transforms import Bbox
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kw = {}
    if width is not None:
        fw = fig.get_figwidth()
        if abs(fw - width * SCALE) > 1e-6:
            raise SystemExit(f"{name}: drawn {fw:.3f} in wide, not {width * SCALE:.3f}")
        # measure at the output resolution: at the screen's 100 dpi, hinting draws text ~9% narrower
        # than the PDF and the 600-dpi PNG, and a label clipped in print passes the check
        dpi0 = fig.dpi
        fig.set_dpi(STYLE["dpi_raster"])
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.set_dpi(dpi0)
        if tb.x0 < -0.001 or tb.x1 > fw + 0.001:          # nothing may be clipped
            raise SystemExit(f"{name}: content runs past the figure's width "
                             f"(x from {tb.x0:.3f} to {tb.x1:.3f} in, figure {fw:.3f} in)")
        pad = STYLE["pad_inches"]
        kw = dict(bbox_inches=Bbox.from_extents(0, max(tb.y0 - pad, 0), fw,
                                                min(tb.y1 + pad, fig.get_figheight())),
                  pad_inches=0)
    for ext in ("pdf", "png"):
        p = OUTPUT / f"{name}.{ext}"
        # 600 dpi for the PNG and for the PDF's rasterized marks (the beeswarm clouds); the PDF's
        # text and lines stay vector whatever the dpi
        fig.savefig(p, dpi=STYLE["dpi_raster"], **kw)
        print(f"wrote {p}", flush=True)
