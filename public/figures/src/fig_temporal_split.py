"""Genera fig_temporal_split.png: partición temporal estricta con embargo.

Fechas reales del dataset (verificadas desde sample_keys de los result.json):
periodo total nov-2024 -> may-2026; test = 16 feb - 8 may 2026 (81 dias).
Conteos del dataset filtrado (56,161): train 39,312 / val 8,424 / test 8,425.
"""

import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = dt.date

# (etiqueta, inicio, fin, color)
spans = [
    ("Train (70%) — 39,312 muestras", D(2024, 11, 23), D(2025, 11, 29), "#d9d9d9"),
    ("Validación (15%) — 8,424 muestras", D(2025, 12, 1), D(2026, 2, 15), "#9ecae1"),
    ("Test (15%) — 8,425 muestras", D(2026, 2, 16), D(2026, 5, 8), "#bcbddc"),
]
embargos = [D(2025, 11, 30), D(2026, 2, 15)]  # fronteras train/val y val/test

fig, ax = plt.subplots(figsize=(9.0, 2.5))
for i, (label, start, end, color) in enumerate(spans):
    y = len(spans) - 1 - i
    ax.barh(y, mdates.date2num(end) - mdates.date2num(start),
            left=mdates.date2num(start), height=0.55, color=color,
            edgecolor="#525252", linewidth=0.8, zorder=3)
    ax.text(mdates.date2num(start), y + 0.42, label, fontsize=9, va="bottom", ha="left")

for x in embargos:
    ax.axvline(mdates.date2num(x), color="#e6550d", linewidth=2.2, zorder=4)

ax.text(mdates.date2num(embargos[0]) + 6, 0.0, "Embargo 24h",
        fontsize=7.5, color="#e6550d", va="center")

ax.set_ylim(-0.6, len(spans) - 0.2)
ax.set_yticks([])
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(mdates.date2num(D(2024, 11, 1)), mdates.date2num(D(2026, 6, 1)))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
ax.set_title("Partición temporal del dataset (56,161 muestras)", fontsize=11, fontweight="bold")

legend = [
    Patch(facecolor="#d9d9d9", edgecolor="#525252", label="Entrenamiento"),
    Patch(facecolor="#9ecae1", edgecolor="#525252", label="Validación"),
    Patch(facecolor="#bcbddc", edgecolor="#525252", label="Prueba"),
    Patch(facecolor="#e6550d", label="Embargo 24h"),
]
ax.legend(handles=legend, loc="lower left", ncol=4, fontsize=7.5, frameon=False,
          bbox_to_anchor=(0.0, -0.28))

fig.tight_layout()
fig.savefig("public/figures/fig_temporal_split.png", dpi=150, bbox_inches="tight")
print("saved public/figures/fig_temporal_split.png")
