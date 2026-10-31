"""Plot per-step traces saved by evaluate.py --trace: whole-body CoM height and sideways torso lean.

  python scripts/plot_traces.py OUT.png TRACE.npz[:label] ...
One column per trace; line = env 0, band = min-max over the traced envs; dashed line = payload placed.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib import font_manager  # noqa: E402

for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
    except (FileNotFoundError, RuntimeError):
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = "#2a78d6"  # single series per panel (reference palette slot 1)

out, specs = sys.argv[1], sys.argv[2:]
fig, axes = plt.subplots(2, len(specs), figsize=(3.2 * len(specs), 5.2), sharex=True, sharey="row", squeeze=False)
fig.patch.set_facecolor(SURF)
for c, spec in enumerate(specs):
    path, _, label = spec.partition(":")
    d = np.load(path)
    t = d["t"][:, 0]
    load = d["load_N"][:, 0]
    on = t[np.argmax(load > 0.5)] if (load > 0.5).any() else None
    for r, (key, ylab) in enumerate([("com_z", "重心の高さ [m]"), ("lean_deg", "胴体の横の傾き [°]")]):
        ax = axes[r, c]
        ax.set_facecolor(SURF)
        y = d[key]
        ax.fill_between(t, y.min(1), y.max(1), color=SERIES, alpha=0.15, linewidth=0)
        ax.plot(t, y[:, 0], color=SERIES, linewidth=1.5)
        if on is not None:
            ax.axvline(on, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            if r == 0:
                ax.text(on + 0.1, 0.02, f"荷物 {float(d['load_kg']):.0f} kg", transform=ax.get_xaxis_transform(),
                        color=MUTED, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        if c == 0:
            ax.set_ylabel(ylab, color=INK, fontsize=9)
        if r == 0:
            ax.set_title(label or path, color=INK, fontsize=10, loc="left")
        if r == 1:
            ax.set_xlabel("時間 [s]", color=MUTED, fontsize=8)
fig.text(0.01, 0.005, "線 = 1台目、帯 = 8台の最小〜最大。二足立ちから開始(最初の約1秒は着地の揺れ)",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, dpi=130, facecolor=SURF)
print("saved", out)
