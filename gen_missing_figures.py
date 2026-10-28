#!/usr/bin/env python3
"""Generate Fig 5 (NAAG binding distance proxy), Fig 11 (MM-PBSA bar)."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, json
from pathlib import Path

mpl.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,
    "axes.titlesize":8,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":8,"lines.linewidth":1.0,"axes.linewidth":0.7,
    "pdf.fonttype":42,"ps.fonttype":42})

OUT   = Path("/home/stalin/Desktop/Zinc/work/results/out")
FIGS  = OUT / "figures"
SKIP  = 5000
TOTAL = 15001
COLORS = {"apo":"#4C72B0","naag":"#DD8452","zno541":"#55A868","tern541":"#C44E52"}
LABELS = {"apo":"Apo","naag":"PSMA-NAAG","zno541":"PSMA-(ZnO)12","tern541":"Ternary"}
DPI = 300
MM = 1/25.4
FULL_W = 174*MM

def load_ts(sys, rep):
    return np.load(OUT/f"{sys}_rep{rep}_timeseries.npz", allow_pickle=True)

# -- Figure 5: NAAG binding stability (d_naag_prot_min proxy for H-bonds) ------
print("Figure 5: NAAG binding distance proxy ...")
fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 5.5*25.4*MM), sharex=True,
                          constrained_layout=True)

for sys in ["naag", "tern541"]:
    for rep in [1, 2]:
        ts = load_ts(sys, rep)
        if "d_naag_prot_min" not in ts.files: continue
        arr = ts["d_naag_prot_min"]
        t   = ts["t_ns"] if "t_ns" in ts.files else np.arange(len(arr))*20/1000
        ax  = axes[0] if sys == "naag" else axes[1]
        lw  = 1.1 if rep == 1 else 0.7
        ls  = "-" if rep == 1 else "--"
        ax.plot(t, arr*10, color=COLORS[sys], lw=lw, ls=ls,
                label=f"{LABELS[sys]} rep{rep}")

for ax, title in zip(axes, ["PSMA-NAAG system", "Ternary (PSMA-NAAG-(ZnO)12) system"]):
    ax.axvline(100, color="gray", lw=0.7, ls=":", label="Production start")
    ax.axhline(3.5, color="red", lw=0.6, ls="--", alpha=0.6, label="H-bond cutoff (3.5 A)")
    ax.set_ylabel("NAAG-protein\nmin distance (A)", fontsize=7)
    ax.set_title(title, fontsize=8, pad=2)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.set_ylim(0, 12)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

axes[1].set_xlabel("Time (ns)", fontsize=8)
axes[0].set_xlim(0, 300)
fig.savefig(str(FIGS/"fig5_naag_distance.pdf"), dpi=DPI, bbox_inches="tight")
fig.savefig(str(FIGS/"fig5_naag_distance.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  Saved fig5_naag_distance.png")

# -- Figure 11: MM-PBSA bar chart -----------------------------------------------
print("Figure 11: MM-PBSA bar chart ...")
mmpbsa_file = Path("/home/stalin/Desktop/Zinc/work/results/mmpbsa/zno541_rep1/mmpbsa_result.json")
mm = json.load(open(mmpbsa_file)) if mmpbsa_file.exists() else {}

terms  = ["dEvdW", "dEelec", "dGPB", "dGnonpol", "dGbind"]
labels = ["ΔEvdW", "ΔEelec", "ΔGPB", "ΔGnonpol", "ΔGbind"]
values = [mm.get("dEvdW_kcal_mean",0), mm.get("dEelec_kcal_mean",0),
          mm.get("dEPB_kcal_mean",0),  mm.get("dEnpol_kcal_mean",0),
          mm.get("dG_bind_kcal_mean",0)]
errors = [0, 0, 0, 0, mm.get("dG_bind_kcal_sd", 0)]
colors_bar = ["#4C72B0" if v < 0 else "#C44E52" for v in values]
colors_bar[-1] = "#DD8452"

fig, ax = plt.subplots(figsize=(FULL_W*0.55, 3.2*25.4*MM), constrained_layout=True)
x = np.arange(len(terms))
bars = ax.bar(x, values, yerr=errors, capsize=3, color=colors_bar,
              width=0.55, edgecolor="0.3", linewidth=0.5, error_kw=dict(lw=0.8))
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Energy (kcal mol-1)", fontsize=8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1 if val >= 0 else bar.get_height()-0.3,
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6.5)
ax.text(0.98, 0.05, "dGbind > 0: unfavourable\n(cage dissociated during MD)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#C44E52",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C44E52", alpha=0.8))
fig.savefig(str(FIGS/"fig11_mmpbsa.pdf"), dpi=DPI, bbox_inches="tight")
fig.savefig(str(FIGS/"fig11_mmpbsa.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  Saved fig11_mmpbsa.png")
print("Done.")
