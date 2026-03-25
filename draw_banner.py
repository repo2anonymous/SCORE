"""
Draw SCORE algorithm banner for GitHub repository.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── palette ──────────────────────────────────────────────────
BG       = "#0d1117"      # GitHub dark bg
WHITE    = "#e6edf3"
GRAY     = "#8b949e"
BLUE     = "#58a6ff"
PURPLE   = "#bc8cff"
GREEN    = "#3fb950"
ORANGE   = "#d29922"
RED      = "#f85149"
CYAN     = "#39d353"
PINK     = "#f778ba"
LIGHT_BG = "#161b22"
BORDER   = "#30363d"

# ── figure ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5.6), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-0.5, 5.6)
ax.set_aspect("equal")
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────
def rounded_box(x, y, w, h, color, label, fontsize=8.5, text_color=WHITE,
                border_color=None, alpha=0.92, bold=False, sublabel=None):
    """Draw a rounded rectangle with centered label."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=border_color,
        linewidth=1.5, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    dy = 0.12 if sublabel else 0
    ax.text(x + w / 2, y + h / 2 + dy, label,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=3,
            fontfamily="monospace")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center", fontsize=6.5,
                color=GRAY, zorder=3, fontfamily="monospace")

def arrow(x1, y1, x2, y2, color=GRAY, style="-|>", lw=1.2, ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, linestyle=ls,
        mutation_scale=12, zorder=4,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(a)

def curved_arrow(x1, y1, x2, y2, color=GRAY, rad=0.3, lw=1.2, ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color,
        linewidth=lw, linestyle=ls,
        mutation_scale=12, zorder=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)

# ── title ────────────────────────────────────────────────────
ax.text(8.25, 5.25, "SCORE", fontsize=26, fontweight="bold",
        color=BLUE, ha="center", va="center", fontfamily="monospace", zorder=5)
ax.text(8.25, 4.82, "Self-diagnostic Critique On-policy Reasoning with Execution feedback",
        fontsize=8, color=GRAY, ha="center", va="center", fontfamily="monospace", zorder=5)

# ══════════════════════════════════════════════════════════════
#  ROW 1 (y≈3.0): Main pipeline
# ══════════════════════════════════════════════════════════════
row1_y = 2.95
bh = 0.85  # box height

# 1) Problem
rounded_box(0.0, row1_y, 1.8, bh, "#1c2333", "Problem", fontsize=9,
            border_color=BORDER, bold=True, sublabel="code task")

# 2) Student Model (π_θ)
rounded_box(2.5, row1_y, 2.2, bh, "#1a2744", "Student  π_θ", fontsize=9,
            border_color=BLUE, bold=True, sublabel="generate code")

# 3) Sandbox
rounded_box(5.5, row1_y, 2.0, bh, "#1a2e1a", "Sandbox", fontsize=9,
            border_color=GREEN, bold=True, sublabel="execute & test")

# 4) Self-Diagnosis (BEACON)
rounded_box(8.3, row1_y, 2.4, bh, "#2d1a3a", "Self-Diagnosis", fontsize=9,
            border_color=PURPLE, bold=True, sublabel="predict error type")

# 5) GRPO Update
rounded_box(11.5, row1_y, 2.0, bh, "#2a2010", "GRPO Update", fontsize=9,
            border_color=ORANGE, bold=True, sublabel="policy gradient")

# 6) Updated π_θ
rounded_box(14.3, row1_y, 1.8, bh, "#1a2744", "Updated π_θ", fontsize=9,
            border_color=BLUE, bold=True)

# ── arrows row 1 ─────────────────────────────────────────────
arrow(1.8, row1_y + bh/2, 2.5, row1_y + bh/2, GRAY)
arrow(4.7, row1_y + bh/2, 5.5, row1_y + bh/2, GRAY)
arrow(7.5, row1_y + bh/2, 8.3, row1_y + bh/2, GRAY)
arrow(10.7, row1_y + bh/2, 11.5, row1_y + bh/2, GRAY)
arrow(13.5, row1_y + bh/2, 14.3, row1_y + bh/2, GRAY)

# ══════════════════════════════════════════════════════════════
#  ROW 2 (y≈1.0): Sub-components
# ══════════════════════════════════════════════════════════════
row2_y = 0.8
sh = 0.75  # smaller box height

# ── 4-Quadrant Calibration ───────────────────────────────────
qx, qy, qw = 5.5, row2_y, 3.6
# background panel
panel = FancyBboxPatch(
    (qx - 0.1, qy - 0.15), qw + 0.2, sh + 0.3,
    boxstyle="round,pad=0.1",
    facecolor=LIGHT_BG, edgecolor=BORDER,
    linewidth=1.0, alpha=0.8, zorder=1,
)
ax.add_patch(panel)
ax.text(qx + qw / 2, qy + sh + 0.03, "BEACON 4-Quadrant",
        fontsize=7, color=PURPLE, ha="center", fontweight="bold",
        fontfamily="monospace", zorder=3)

cell_w, cell_h = 0.82, 0.3
gap = 0.08
ox = qx + (qw - 2 * cell_w - gap) / 2
oy = qy

quads = [
    # (col, row, label, color)
    (0, 1, "A  Pass✓ Pred✓", "#1a3a1a"),   # top-left
    (1, 1, "C  Pass✓ Pred✗", "#2a2a10"),   # top-right
    (0, 0, "B  Fail✗ Pred✓", "#3a1a1a"),   # bottom-left  BLINDSPOT
    (1, 0, "D  Fail✗ Pred✗", "#1c2333"),   # bottom-right
]
for col, row, label, c in quads:
    bx = ox + col * (cell_w + gap)
    by = oy + row * (cell_h + gap)
    box = FancyBboxPatch(
        (bx, by), cell_w, cell_h,
        boxstyle="round,pad=0.04",
        facecolor=c, edgecolor=BORDER,
        linewidth=0.8, alpha=0.95, zorder=2,
    )
    ax.add_patch(box)
    fc = RED if "B " in label else (GREEN if "A " in label else (ORANGE if "C " in label else GRAY))
    ax.text(bx + cell_w / 2, by + cell_h / 2, label,
            ha="center", va="center", fontsize=5.5,
            color=fc, fontfamily="monospace", zorder=3)

# "BLINDSPOT" label on B
bx_b = ox
by_b = oy
ax.text(bx_b + cell_w / 2, by_b - 0.11, "↑ blindspot",
        fontsize=5, color=RED, ha="center", fontfamily="monospace",
        fontweight="bold", zorder=3)

# arrow from Self-Diagnosis down to BEACON
arrow(9.5, row1_y, 8.0, row2_y + sh + 0.18, PURPLE, lw=1.0, ls="--")

# ── EMA Teacher + Distillation ───────────────────────────────
rounded_box(10.0, row2_y, 2.2, sh, "#2d1a3a", "EMA Teacher", fontsize=8,
            border_color=PINK, bold=True, sublabel="T=(1-α)T + αS")

rounded_box(12.8, row2_y, 2.4, sh, "#2a2010", "KL Distillation", fontsize=8,
            border_color=ORANGE, bold=True, sublabel="top-K + IS clip")

# arrow from EMA → Distillation
arrow(12.2, row2_y + sh / 2, 12.8, row2_y + sh / 2, PINK)

# arrow from Distillation up to GRPO Update
arrow(14.0, row2_y + sh, 12.8, row1_y, ORANGE, lw=1.0, ls="--")

# arrow from GRPO Update down to EMA Teacher
curved_arrow(12.0, row1_y, 11.1, row2_y + sh, PINK, rad=-0.2, lw=1.0, ls="--")

# arrow from BEACON to Distillation (blindspot weights)
arrow(9.2, row2_y + sh / 2, 10.0, row2_y + sh / 2, PURPLE, lw=1.0, ls="--")

# ── reward label ─────────────────────────────────────────────
ax.text(6.5, row1_y - 0.18, "reward + error type",
        fontsize=6, color=GREEN, ha="center", fontfamily="monospace", zorder=3)
arrow(6.5, row1_y, 6.5, row2_y + sh + 0.18, GREEN, lw=1.0, ls="--")

# ── loop arrow from Updated π_θ back to Student ─────────────
curved_arrow(15.2, row1_y, 15.2, 4.6, BLUE, rad=-0.4, lw=1.0, ls="--")
curved_arrow(15.2, 4.6, 3.6, 4.6, BLUE, rad=0.0, lw=1.0, ls="--")
curved_arrow(3.6, 4.6, 3.6, row1_y + bh, BLUE, rad=-0.0, lw=1.0, ls="--")
ax.text(9.4, 4.52, "next iteration", fontsize=6.5, color=BLUE,
        ha="center", fontfamily="monospace", zorder=3)

# ── legend / key equations ───────────────────────────────────
eqs = [
    ("W_B = w₀ · (1 + blindspot_ema)", PURPLE),
    ("L = L_GRPO + β · L_distill", ORANGE),
    ("CCS = 1 - E|p̂ - p|", CYAN),
]
for i, (eq, c) in enumerate(eqs):
    ax.text(0.2, 1.95 - i * 0.3, eq, fontsize=6.5, color=c,
            fontfamily="monospace", zorder=3)

# ── save ─────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
fig.savefig("images/banner.png", dpi=200, facecolor=BG,
            bbox_inches="tight", pad_inches=0.15)
print("Saved to images/banner.png")
plt.close()
