"""Vector schematic figures for the manuscript (drawn, not AI-generated, so
every label is exact and editable). Writes PNG + PDF to
results/v3/analysis/figures/.

Run from the repo root:  python scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "v3" / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8984"
FILL, ACCENT, ACCENT_FILL = "#f3f2ef", "#2a78d6", "#e3eefb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, body="", accent=False, title_size=7.4, body_size=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                lw=1.0, ec=ACCENT if accent else LINE,
                                fc=ACCENT_FILL if accent else FILL))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=body_size,
                color=INK2, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=INK2, shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------- pipeline overview
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlim(-0.012, 1.012)
ax.set_ylim(0, 1)
ax.axis("off")

W, H, Y = 0.176, 0.50, 0.47
xs = [0.0, 0.206, 0.412, 0.618, 0.824]
stages = [
    ("1  Data", "Cotton Disease\nDataset (Kaggle)\n2,310 images\n4 classes\n\nOriginal split:\n1,951 / 253 / 106\nDe-duplicated split:\n1,946 / 236 / 105"),
    ("2  Preprocessing", "Resize 224 × 224\nbackbone-specific\nnormalisation\n\nTrain-only\naugmentation:\nrotate, shift, shear,\nzoom, flip, brightness"),
    ("3  Models", "MobileNetV3Small\nEfficientNetB0\nDenseNet121\n(ImageNet weights)\n\n+ identical\nclassifier head\n(Fig. 2)"),
    ("4  Training", "Phase 1: head only\nPhase 2: top 30 %\nunfrozen, BatchNorm\nfrozen; new callbacks;\nbetter phase kept\n\n2 splits × 3 models ×\n5 seeds = 30 runs"),
    ("5  Evaluation", "Accuracy, macro-F1,\nbalanced acc., MCC, κ\n\nhierarchical\nbootstrap CIs\nimage-level paired\ntests (Holm), GEE\nodds ratios"),
]
for i, (x, (t, b)) in enumerate(zip(xs, stages)):
    box(ax, x, Y, W, H, t, b, accent=(i == 3))
    if i < len(xs) - 1:
        arrow(ax, x + W + 0.006, Y + H / 2, xs[i + 1] - 0.006, Y + H / 2)

LW, LH, LY = 0.31, 0.29, 0.03
low = [
    (0.0, "Duplicate audit", "pHash of all 2,310 images; Hamming\ndistance ≤ 5 → duplicate groups;\nde-duplicated split keeps test copy;\npaired original vs de-duplicated"),
    (0.345, "Explainability", "Grad-CAM on the final feature map;\ndeletion/insertion-AUC in pixel\nspace (black / blurred baseline);\nall test images, seeds and splits"),
    (0.69, "Efficiency", "Parameters and weight size;\nsingle-image CPU latency\n(graph mode, one session);\nTFLite float32 and int8"),
]
for x, t, b in low:
    box(ax, x, LY, LW, LH, t, b)
arrow(ax, xs[0] + W / 2, Y - 0.006, 0.0 + LW / 2, LY + LH + 0.006)
arrow(ax, xs[3] + W / 2, Y - 0.006, 0.345 + LW / 2, LY + LH + 0.006)
arrow(ax, xs[4] + W / 2, Y - 0.006, 0.69 + LW / 2, LY + LH + 0.006)
save(fig, "fig_pipeline")

# ----------------------------------------------- architecture + two phases
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.set_xlim(-0.012, 1.012)
ax.set_ylim(0, 1)
ax.axis("off")

yb, hb = 0.50, 0.34
ax.add_patch(Rectangle((0.0, yb + 0.05), 0.085, 0.24, fc=FILL, ec=LINE, lw=1))
ax.text(0.0425, yb + 0.17, "Input\n224 × 224\n× 3", ha="center", va="center", fontsize=6.4, color=INK)
arrow(ax, 0.089, yb + hb / 2, 0.106, yb + hb / 2)

bx, bw = 0.11, 0.40
ax.add_patch(Rectangle((bx, yb), bw * 0.7, hb, fc="#e4e3df", lw=1, hatch="////", edgecolor=LINE))
ax.add_patch(Rectangle((bx + bw * 0.7, yb), bw * 0.3, hb, fc=ACCENT_FILL, ec=ACCENT, lw=1.2))
ax.text(bx + bw * 0.35, yb + hb + 0.03, "Bottom 70 % of backbone layers (always frozen)",
        ha="center", fontsize=6.4, color=INK2)
ax.text(bx + bw * 0.85, yb - 0.05, "Top 30 % of layers", ha="center", fontsize=6.4, color=ACCENT)
ax.text(bx + bw * 0.35, yb + hb / 2,
        "ImageNet-pretrained backbone\nMobileNetV3Small /\nEfficientNetB0 / DenseNet121",
        ha="center", va="center", fontsize=6.4, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.92))
ax.text(bx + bw * 0.85, yb + hb / 2, "frozen in\nphase 1;\ntrainable in\nphase 2\n(BatchNorm\nstays frozen)",
        ha="center", va="center", fontsize=6.2, color=ACCENT)

hx = 0.54
head = ["Global\naverage\npooling", "Dropout\n0.3", "Dense\n128\nReLU", "Dropout\n0.2", "Dense 4\nsoftmax"]
hw, gap = 0.076, 0.017
arrow(ax, bx + bw + 0.004, yb + hb / 2, hx - 0.004, yb + hb / 2)
for i, lab in enumerate(head):
    x = hx + i * (hw + gap)
    ax.add_patch(FancyBboxPatch((x, yb + 0.04), hw, hb - 0.08,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=ACCENT_FILL, ec=ACCENT, lw=1))
    ax.text(x + hw / 2, yb + hb / 2, lab, ha="center", va="center", fontsize=6.0, color=INK)
    if i < len(head) - 1:
        arrow(ax, x + hw + 0.002, yb + hb / 2, x + hw + gap - 0.002, yb + hb / 2)
ax.text(hx + (5 * hw + 4 * gap) / 2, yb + hb + 0.03,
        "Classifier head (identical for all backbones, trainable in both phases)",
        ha="center", fontsize=6.4, color=INK2)

ty = 0.13
ax.text(0.0, ty + 0.075, "Schedule", ha="left", va="center", fontsize=7, fontweight="bold", color=INK)
ax.add_patch(Rectangle((0.11, ty), 0.36, 0.15, fc=FILL, ec=LINE, lw=1))
ax.add_patch(Rectangle((0.47, ty), 0.525, 0.15, fc=ACCENT_FILL, ec=ACCENT, lw=1))
ax.text(0.29, ty + 0.075, "Phase 1 — head only (backbone frozen)\nAdam, lr 1×10⁻³, ≤ 20 epochs",
        ha="center", va="center", fontsize=6.6, color=INK)
ax.text(0.7325, ty + 0.075, "Phase 2 — fine-tune top 30 % of backbone\nAdam, lr 1×10⁻⁵, ≤ 18 epochs; final model =\nphase with the lower validation loss",
        ha="center", va="center", fontsize=6.6, color=INK)
ax.text(0.5, 0.04,
        "Both phases: class-weighted cross-entropy · new early-stopping (validation loss, patience 5, "
        "restore best) and reduce-LR-on-plateau (×0.5, patience 2) instances per phase",
        ha="center", fontsize=6.1, color=INK2)
save(fig, "fig_architecture")
print("wrote fig_pipeline and fig_architecture")
