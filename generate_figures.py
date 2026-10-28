#!/usr/bin/env python3
"""
generate_figures.py
-------------------
Generate all manuscript figures for the PSMA-(ZnO)12 MD paper.

Input:  out/{sys}_rep{N}_timeseries.npz
        out/{sys}_rep{N}_summary.json
        out/{sys}_rep{N}_energy.npz

Output: out/figures/fig{N}.pdf  +  out/figures/fig{N}.png

Run after psma_md_analysis.py has finished for all systems / replicas.
"""

# ─────────────────────────── imports ───────────────────────────────────────
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# ─────────────────────────── global style ──────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.7",
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    }
)

# mm → inch conversion
MM_TO_IN = 1 / 25.4
FULL_W = 180 * MM_TO_IN       # full journal column (180 mm)
HALF_W = 90 * MM_TO_IN        # single column (90 mm)
DPI = 300

# ─────────────────────────── colour scheme ─────────────────────────────────
SYSTEMS = ["apo", "naag", "zno541", "tern541"]
LABELS  = {"apo": "Apo", "naag": "NAAG", "zno541": "(ZnO)₁₂", "tern541": "Tern."}
COLORS  = {"apo": "#808080", "naag": "#2166ac", "zno541": "#1a9641", "tern541": "#d73027"}
REPS    = [1, 2]

# simulation parameters
DT_PS      = 20.0        # ps per frame
SKIP_NS    = 100.0       # equilibration to discard
TOTAL_NS   = 300.0
# frame indices
SKIP_FRAME = int(SKIP_NS * 1000 / DT_PS)   # 5000
TOTAL_FRAMES = int(TOTAL_NS * 1000 / DT_PS) + 1  # 15001

# ─────────────────────────── path helpers ──────────────────────────────────
HERE     = Path(__file__).parent
OUT_DIR  = HERE / "out"
FIG_DIR  = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def ts_path(sys: str, rep: int) -> Path:
    return OUT_DIR / f"{sys}_rep{rep}_timeseries.npz"

def sm_path(sys: str, rep: int) -> Path:
    return OUT_DIR / f"{sys}_rep{rep}_summary.json"

def en_path(sys: str, rep: int) -> Path:
    return OUT_DIR / f"{sys}_rep{rep}_energy.npz"


def load_ts(sys: str, rep: int) -> dict | None:
    p = ts_path(sys, rep)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def load_sm(sys: str, rep: int) -> dict | None:
    p = sm_path(sys, rep)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def load_en(sys: str, rep: int) -> dict | None:
    p = en_path(sys, rep)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


# ─────────────────────────── generic helpers ───────────────────────────────

def production_mask(t_ns: np.ndarray) -> np.ndarray:
    """Boolean mask: frames with t >= SKIP_NS."""
    return t_ns >= SKIP_NS


def get_production(arr: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    return arr[production_mask(t_ns)]


def smooth(y: np.ndarray, window: int = 250) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def mean_over_reps(key: str, sys: str, t_ref: np.ndarray | None = None) -> tuple:
    """
    Returns (t_ns, mean_arr, std_arr, [arr_rep1, arr_rep2]).
    If only one replica exists the std is zero.
    """
    arrs, ts = [], []
    for rep in REPS:
        d = load_ts(sys, rep)
        if d is None or key not in d:
            continue
        arr = d[key]
        t   = d["t_ns"] if "t_ns" in d else np.arange(len(arr)) * DT_PS / 1000.0
        arrs.append(arr)
        ts.append(t)
    if not arrs:
        return None, None, None, []
    # interpolate to common time axis if needed
    if t_ref is None:
        t_ref = ts[0]
    aligned = []
    for a, t in zip(arrs, ts):
        if len(a) == len(t_ref) and np.allclose(t, t_ref):
            aligned.append(a)
        else:
            aligned.append(np.interp(t_ref, t, a))
    stack = np.vstack(aligned)
    return t_ref, stack.mean(0), stack.std(0, ddof=min(1, len(aligned) - 1)), aligned


def production_values(key: str, sys: str) -> list:
    """Collect all production-window values across both replicas (flat)."""
    vals = []
    for rep in REPS:
        d = load_ts(sys, rep)
        if d is None or key not in d:
            continue
        t = d["t_ns"] if "t_ns" in d else np.arange(len(d[key])) * DT_PS / 1000.0
        v = get_production(d[key], t)
        vals.extend(v[~np.isnan(v)].tolist())
    return vals


def violin_box_panel(ax, data_dict: dict, ylabel: str, color_map: dict | None = None):
    """
    Draw violin + box plots for each system that has data.
    data_dict: {sys_name: 1-D array-like}
    """
    if color_map is None:
        color_map = COLORS
    keys   = [k for k in SYSTEMS if k in data_dict and len(data_dict[k]) > 0]
    values = [np.asarray(data_dict[k]) for k in keys]
    cols   = [color_map[k] for k in keys]

    positions = np.arange(1, len(keys) + 1)
    parts = ax.violinplot(values, positions=positions, showmedians=False,
                          showextrema=False, widths=0.7)
    for body, c in zip(parts["bodies"], cols):
        body.set_facecolor(c)
        body.set_alpha(0.45)
        body.set_edgecolor(c)

    bp = ax.boxplot(values, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[k] for k in keys], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", which="both", labelsize=7)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def save_fig(fig, name: str):
    """Save as both PDF and PNG."""
    stem = FIG_DIR / name
    fig.savefig(str(stem) + ".pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=DPI, bbox_inches="tight")
    print(f"  saved: {stem}.pdf  +  .png")


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


# ─────────────────────────── Figure 2 – RMSD ───────────────────────────────

def figure2_rmsd():
    print("Figure 2: RMSD …")
    key = "rmsd_bb"
    # build common time axis from first available replica
    t_ref = None
    for sys in SYSTEMS:
        for rep in REPS:
            d = load_ts(sys, rep)
            if d and "t_ns" in d:
                t_ref = d["t_ns"]
                break
        if t_ref is not None:
            break
    if t_ref is None:
        print("  [skip] no timeseries data found.")
        return

    # figure layout: 4 time-series panels (left) + 1 violin (right)
    fig = plt.figure(figsize=(FULL_W, 5.2 * MM_TO_IN * 25))
    gs  = gridspec.GridSpec(4, 2, width_ratios=[3, 1], hspace=0.08, wspace=0.35,
                            left=0.09, right=0.97, top=0.94, bottom=0.08)

    vdata = {}
    for row, sys in enumerate(SYSTEMS):
        ax = fig.add_subplot(gs[row, 0])
        t, mn, sd, reps_data = mean_over_reps(key, sys, t_ref)
        if t is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_yticks([])
        else:
            # thin individual replicas
            for i, arr in enumerate(reps_data):
                ax.plot(t, arr, color=COLORS[sys], lw=0.5, alpha=0.4,
                        label=f"Rep {i+1}" if row == 0 else None)
            # mean ± SD shading
            ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.25)
            ax.plot(t, mn, color=COLORS[sys], lw=1.4, label="Mean" if row == 0 else None)
            prod = get_production(mn, t)
            vdata[sys] = prod

            # mark skip
            ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
            ax.set_xlim(0, TOTAL_NS)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        label_str = f"RMSD (nm)" if row == 2 else ""
        ax.set_ylabel("RMSD (nm)", fontsize=8)
        ax.text(0.99, 0.93, LABELS[sys], transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=COLORS[sys], fontweight="bold")
        if row < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (ns)", fontsize=8)
        add_panel_label(ax, chr(65 + row))

    # legend on first panel
    ax0 = fig.axes[0]
    ax0.legend(fontsize=7, loc="upper left", ncol=3)

    # right panel: violin+box
    ax_vio = fig.add_subplot(gs[:, 1])
    # collect per-system production values
    violin_box_panel(ax_vio, vdata, "Backbone RMSD (nm)")
    add_panel_label(ax_vio, "E", x=-0.18)

    save_fig(fig, "fig2_rmsd")
    plt.close(fig)


# ─────────────────────────── Figure 3 – RMSF ───────────────────────────────

def figure3_rmsf():
    print("Figure 3: RMSF …")
    fig, axes = plt.subplots(4, 1, figsize=(FULL_W, 7.0 * MM_TO_IN * 25),
                             sharex=True, constrained_layout=True)
    for ax, sys in zip(axes, SYSTEMS):
        arrs, stds, resids_ref = [], [], None
        for rep in REPS:
            d = load_ts(sys, rep)
            if d is None or "rmsf_ca" not in d:
                continue
            arrs.append(d["rmsf_ca"])
            stds.append(d["rmsf_ca_blockstd"])
            if resids_ref is None and "rmsf_resids" in d:
                resids_ref = d["rmsf_resids"]
        if not arrs:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        mn = np.mean(arrs, axis=0)
        # combined block-std: sqrt(mean of variances)
        combined_std = np.sqrt(np.mean([s**2 for s in stds], axis=0))
        x = resids_ref if resids_ref is not None else np.arange(len(mn))

        ax.fill_between(x, mn - combined_std, mn + combined_std,
                        color=COLORS[sys], alpha=0.3)
        ax.plot(x, mn, color=COLORS[sys], lw=1.1, label=LABELS[sys])
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_ylabel("RMSF (nm)", fontsize=8)
        ax.text(0.99, 0.96, LABELS[sys], transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=COLORS[sys], fontweight="bold")
        add_panel_label(ax, chr(65 + SYSTEMS.index(sys)))

    axes[-1].set_xlabel("Residue index", fontsize=8)
    save_fig(fig, "fig3_rmsf")
    plt.close(fig)


# ─────────────────────────── Figure 4 – Rg ─────────────────────────────────

def figure4_rg():
    print("Figure 4: Radius of gyration …")
    key = "rg_prot"
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8 * MM_TO_IN * 25),
                             constrained_layout=True)
    ax_ts, ax_vio = axes

    t_ref = None
    for sys in SYSTEMS:
        for rep in REPS:
            d = load_ts(sys, rep)
            if d and "t_ns" in d:
                t_ref = d["t_ns"]
                break
        if t_ref is not None:
            break

    vdata = {}
    for sys in SYSTEMS:
        t, mn, sd, reps_data = mean_over_reps(key, sys, t_ref)
        if t is None:
            continue
        for arr in reps_data:
            ax_ts.plot(t, arr, color=COLORS[sys], lw=0.5, alpha=0.3)
        ax_ts.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
        ax_ts.plot(t, mn, color=COLORS[sys], lw=1.3, label=LABELS[sys])
        vdata[sys] = get_production(mn, t)

    ax_ts.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax_ts.set_xlabel("Time (ns)", fontsize=8)
    ax_ts.set_ylabel("Rg (nm)", fontsize=8)
    ax_ts.legend(fontsize=8)
    ax_ts.set_xlim(0, TOTAL_NS)
    ax_ts.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax_ts, "A")

    violin_box_panel(ax_vio, vdata, "Rg (nm)")
    add_panel_label(ax_vio, "B", x=-0.18)

    save_fig(fig, "fig4_rg")
    plt.close(fig)


# ─────────────────────────── Figure 5 – SASA (protein) ────────────────────

def figure5_sasa():
    print("Figure 5: Protein SASA …")
    key = "sasa_prot_nm2"
    # SASA is computed on a sub-sampled grid; need special time axis
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8 * MM_TO_IN * 25),
                             constrained_layout=True)
    ax_ts, ax_vio = axes
    vdata = {}

    for sys in SYSTEMS:
        t_arrs, v_arrs = [], []
        for rep in REPS:
            d = load_ts(sys, rep)
            if d is None or key not in d:
                continue
            frames = d["sasa_frames"] if "sasa_frames" in d else None
            t_base = d["t_ns"] if "t_ns" in d else None
            if frames is not None and t_base is not None:
                t_s = t_base[frames.astype(int)]
            elif t_base is not None:
                # assume same length
                t_s = t_base[np.linspace(0, len(t_base)-1, len(d[key])).astype(int)]
            else:
                t_s = np.arange(len(d[key])) * DT_PS / 1000.0
            t_arrs.append(t_s)
            v_arrs.append(d[key])

        if not v_arrs:
            continue
        # align to shortest
        min_len = min(len(v) for v in v_arrs)
        t_comm  = t_arrs[0][:min_len]
        v_stack = np.vstack([v[:min_len] for v in v_arrs])
        mn, sd  = v_stack.mean(0), v_stack.std(0, ddof=min(1, len(v_arrs)-1))

        for v in v_arrs:
            ax_ts.plot(t_arrs[0][:len(v)], v, color=COLORS[sys], lw=0.5, alpha=0.3)
        ax_ts.fill_between(t_comm, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
        ax_ts.plot(t_comm, mn, color=COLORS[sys], lw=1.3, label=LABELS[sys])

        mask_prod = t_comm >= SKIP_NS
        vdata[sys] = mn[mask_prod]

    ax_ts.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax_ts.set_xlabel("Time (ns)", fontsize=8)
    ax_ts.set_ylabel("Protein SASA (nm²)", fontsize=8)
    ax_ts.legend(fontsize=8)
    ax_ts.set_xlim(0, TOTAL_NS)
    ax_ts.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax_ts, "A")

    violin_box_panel(ax_vio, vdata, "Protein SASA (nm²)")
    add_panel_label(ax_vio, "B", x=-0.18)

    save_fig(fig, "fig5_sasa")
    plt.close(fig)


# ─────────────────────────── Figure 6 – Trp SASA ──────────────────────────

def figure6_trp_sasa():
    print("Figure 6: Trp SASA …")
    # discover Trp residues present in the data
    trp_resids_all = set()
    for sys in SYSTEMS:
        for rep in REPS:
            d = load_ts(sys, rep)
            if d is None:
                continue
            for k in d.keys():
                if k.startswith("sasa_trp") and k.endswith("_A2"):
                    try:
                        trp_resids_all.add(int(k.split("sasa_trp")[1].split("_A2")[0]))
                    except ValueError:
                        pass
    if not trp_resids_all:
        print("  [skip] no Trp SASA keys found in any timeseries.")
        return

    # Preferred residues from the spec: 246, 410, 541; show all if <= 6 total,
    # otherwise restrict to preferred + any missing from that list that exist
    PREFERRED = [246, 410, 541]
    all_sorted = sorted(trp_resids_all)
    if len(all_sorted) <= 6:
        trp_resids = all_sorted
    else:
        trp_resids = sorted(
            [r for r in PREFERRED if r in trp_resids_all]
            + [r for r in all_sorted if r not in PREFERRED][:max(0, 6 - len(PREFERRED))]
        )
    if not trp_resids:
        trp_resids = all_sorted[:6]

    n_trp  = len(trp_resids)
    n_sys  = len(SYSTEMS)

    row_h  = max(1.8, 1.8)   # inches per row
    fig, axes = plt.subplots(n_trp, n_sys,
                             figsize=(FULL_W, n_trp * row_h),
                             sharex=True, constrained_layout=True)
    if n_trp == 1:
        axes = axes[np.newaxis, :]
    if n_sys == 1:
        axes = axes[:, np.newaxis]

    for col, sys in enumerate(SYSTEMS):
        for row, resid in enumerate(trp_resids):
            ax = axes[row, col]
            key = f"sasa_trp{resid}_A2"
            t_arrs, v_arrs = [], []
            for rep in REPS:
                d = load_ts(sys, rep)
                if d is None or key not in d:
                    continue
                frames = d["sasa_frames"] if "sasa_frames" in d else None
                t_base = d["t_ns"] if "t_ns" in d else None
                if frames is not None and t_base is not None:
                    t_s = t_base[frames.astype(int)]
                else:
                    t_s = np.arange(len(d[key])) * DT_PS / 1000.0
                t_arrs.append(t_s)
                v_arrs.append(d[key])

            if not v_arrs:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center",
                        color="0.6", fontsize=10)
                ax.set_yticks([])
            else:
                min_len = min(len(v) for v in v_arrs)
                t_comm  = t_arrs[0][:min_len]
                v_stack = np.vstack([v[:min_len] for v in v_arrs])
                mn = v_stack.mean(0)
                sd = v_stack.std(0, ddof=min(1, len(v_arrs)-1))

                for v in v_arrs:
                    ax.plot(t_arrs[0][:len(v)], v, color=COLORS[sys], lw=0.4, alpha=0.35)
                ax.fill_between(t_comm, mn - sd, mn + sd, color=COLORS[sys], alpha=0.25)
                ax.plot(t_comm, mn, color=COLORS[sys], lw=1.2)
                ax.axvline(SKIP_NS, color="gray", lw=0.6, ls="--")
                ax.set_xlim(0, TOTAL_NS)
                ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

            if row == 0:
                ax.set_title(LABELS[sys], fontsize=8, color=COLORS[sys], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Trp{resid}\nSASA (Å²)", fontsize=7)
            if row == n_trp - 1:
                ax.set_xlabel("Time (ns)", fontsize=8)

    save_fig(fig, "fig6_trp_sasa")
    plt.close(fig)


# ─────────────────────────── Figure 7 – H-bonds ───────────────────────────

def figure7_hbonds():
    print("Figure 7: H-bond profiles …")
    # look for hbond keys in any timeseries npz
    hbond_keys = set()
    for sys in SYSTEMS:
        for rep in REPS:
            d = load_ts(sys, rep)
            if d:
                for k in d.keys():
                    if "hbond" in k.lower() or "hb_" in k.lower():
                        hbond_keys.add(k)

    if not hbond_keys:
        print("  [skip] no hbond keys found in timeseries files.")
        return

    hbond_keys = sorted(hbond_keys)
    fig, axes = plt.subplots(len(hbond_keys), 1,
                             figsize=(FULL_W, len(hbond_keys) * 2.2 * MM_TO_IN * 25),
                             sharex=True, constrained_layout=True)
    if len(hbond_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, hbond_keys):
        for sys in SYSTEMS:
            t, mn, sd, _ = mean_over_reps(key, sys)
            if t is None:
                continue
            ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
            ax.plot(t, mn, color=COLORS[sys], lw=1.1, label=LABELS[sys])
        ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
        ax.set_ylabel(key.replace("_", " "), fontsize=7)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_xlim(0, TOTAL_NS)

    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (ns)", fontsize=8)
    save_fig(fig, "fig7_hbonds")
    plt.close(fig)


# ─────────────────────────── Figure 8 – Distances ─────────────────────────

def figure8_distances():
    print("Figure 8: Minimum-distance profiles …")
    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 6.5 * MM_TO_IN * 25),
                             sharex=True, constrained_layout=True)

    # (a) ZnO–protein min distance – zno541, tern541
    ax = axes[0]
    key = "d_zno_prot_min"
    has_data = False
    for sys in ["zno541", "tern541"]:
        t, mn, sd, reps_data = mean_over_reps(key, sys)
        if t is None:
            continue
        has_data = True
        ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.25)
        ax.plot(t, mn, color=COLORS[sys], lw=1.2, label=LABELS[sys])
    if has_data:
        ax.axhline(0.45, color="0.4", lw=0.7, ls=":", label="0.45 nm (bound)")
        ax.axhline(0.80, color="0.7", lw=0.7, ls=":")
    ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("(ZnO)₁₂–protein\nmin dist (nm)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, TOTAL_NS)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax, "A")

    # (b) Trp541 indole CoM to ZnO CoM
    ax = axes[1]
    key = "d_zno_indole_com"
    has_data = False
    for sys in ["zno541", "tern541"]:
        t, mn, sd, _ = mean_over_reps(key, sys)
        if t is None:
            continue
        has_data = True
        ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.25)
        ax.plot(t, mn, color=COLORS[sys], lw=1.2, label=LABELS[sys])
    if not has_data:
        ax.text(0.5, 0.5, "ZnO data not available", transform=ax.transAxes,
                ha="center", color="0.5")
    ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Trp541 indole – (ZnO)₁₂\nCoM dist (nm)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax, "B")

    # (c) NAAG–protein min distance – naag, tern541
    ax = axes[2]
    key = "d_naag_prot_min"
    has_data = False
    for sys in ["naag", "tern541"]:
        t, mn, sd, _ = mean_over_reps(key, sys)
        if t is None:
            continue
        has_data = True
        ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.25)
        ax.plot(t, mn, color=COLORS[sys], lw=1.2, label=LABELS[sys])
    if not has_data:
        ax.text(0.5, 0.5, "NAAG data not available", transform=ax.transAxes,
                ha="center", color="0.5")
    ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Time (ns)", fontsize=8)
    ax.set_ylabel("NAAG–protein\nmin dist (nm)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax, "C")

    save_fig(fig, "fig8_distances")
    plt.close(fig)


# ─────────────────────────── Figure 9 – FRET ──────────────────────────────

def figure9_fret():
    print("Figure 9: FRET distance analysis …")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8 * MM_TO_IN * 25),
                             constrained_layout=True)
    ax_ts, ax_pr, ax_fb = axes

    r_key = "d_zno_indole_com"
    r_fret_nm = 3.0   # approximate Förster radius for Trp–ZnO (adjust if known)

    # (a) time series of r in zno541 and tern541
    for sys in ["zno541", "tern541"]:
        t, mn, sd, reps_data = mean_over_reps(r_key, sys)
        if t is None:
            continue
        for arr in reps_data:
            ax_ts.plot(t, arr, color=COLORS[sys], lw=0.5, alpha=0.35)
        ax_ts.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
        ax_ts.plot(t, mn, color=COLORS[sys], lw=1.3, label=LABELS[sys])

    ax_ts.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax_ts.axhline(r_fret_nm, color="purple", lw=0.8, ls="--",
                  label=f"R₀ ≈ {r_fret_nm} nm")
    ax_ts.set_xlabel("Time (ns)", fontsize=8)
    ax_ts.set_ylabel("Trp541 – (ZnO)₁₂ CoM (nm)", fontsize=8)
    ax_ts.legend(fontsize=7)
    ax_ts.set_xlim(0, TOTAL_NS)
    ax_ts.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax_ts, "A")

    # (b) P(r) distributions + 1/r^6 FRET efficiency overlay
    r_grid = np.linspace(0.01, 6.0, 300)
    E_fret  = 1.0 / (1.0 + (r_grid / r_fret_nm) ** 6)
    ax2b = ax_pr.twinx()

    for sys in ["zno541", "tern541"]:
        # try to get P_r from summary json first
        pr_data = None
        for rep in REPS:
            sm = load_sm(sys, rep)
            if sm and "P_r_indole_com" in sm:
                pr_data = sm["P_r_indole_com"]
                break
        # fallback: reconstruct from timeseries
        if pr_data is None:
            vals = production_values(r_key, sys)
            if vals:
                h, e = np.histogram(vals, bins=np.arange(0, 6.01, 0.05), density=True)
                r_c  = (e[1:] + e[:-1]) / 2
                pr_data = {"r_nm": r_c.tolist(), "P": h.tolist()}

        if pr_data:
            r_nm = np.asarray(pr_data["r_nm"])
            P    = np.asarray(pr_data["P"])
            ax_pr.plot(r_nm, P, color=COLORS[sys], lw=1.3, label=LABELS[sys])
            ax_pr.fill_between(r_nm, P, color=COLORS[sys], alpha=0.2)

    ax2b.plot(r_grid, E_fret, color="purple", lw=1.0, ls="--", label="E(r) FRET")
    ax2b.set_ylabel("FRET efficiency E(r)", fontsize=7, color="purple")
    ax2b.tick_params(axis="y", colors="purple", labelsize=7)
    ax2b.set_ylim(0, 1.05)

    ax_pr.set_xlabel("r (nm)", fontsize=8)
    ax_pr.set_ylabel("P(r)", fontsize=8)
    ax_pr.legend(fontsize=7)
    ax_pr.set_xlim(0, 6)
    add_panel_label(ax_pr, "B", x=-0.18)

    # (c) fraction bound vs distance threshold
    thresholds = np.arange(0.1, 3.05, 0.05)
    for sys in ["zno541", "tern541"]:
        vals = np.asarray(production_values(r_key, sys))
        if len(vals) == 0:
            continue
        fb = np.array([(vals < thr).mean() for thr in thresholds])
        ax_fb.plot(thresholds, fb, color=COLORS[sys], lw=1.3, label=LABELS[sys])

    ax_fb.set_xlabel("Distance threshold (nm)", fontsize=8)
    ax_fb.set_ylabel("Fraction bound", fontsize=8)
    ax_fb.set_ylim(0, 1.05)
    ax_fb.legend(fontsize=7)
    ax_fb.axvline(0.45, color="0.4", lw=0.7, ls=":")
    ax_fb.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax_fb, "C", x=-0.15)

    save_fig(fig, "fig9_fret")
    plt.close(fig)


# ─────────────────────────── Figure 10 – Zn coordination ──────────────────

def figure10_zn_coordination():
    print("Figure 10: Zn active-site coordination …")
    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 7.0 * MM_TO_IN * 25),
                             sharex=False, constrained_layout=True)

    # (a) Zn–Zn distance
    ax = axes[0]
    t_ref = None
    for sys in SYSTEMS:
        for rep in REPS:
            d = load_ts(sys, rep)
            if d and "t_ns" in d:
                t_ref = d["t_ns"]
                break
        if t_ref is not None:
            break

    for sys in SYSTEMS:
        t, mn, sd, _ = mean_over_reps("zn_zn", sys, t_ref)
        if t is None or np.all(np.isnan(mn)):
            continue
        ax.fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
        ax.plot(t, mn, color=COLORS[sys], lw=1.2, label=LABELS[sys])
    ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Zn–Zn distance (nm)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, TOTAL_NS)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    add_panel_label(ax, "A")

    # (b) coordination numbers
    ax = axes[1]
    for sys in SYSTEMS:
        t, mn1, sd1, _ = mean_over_reps("ncoord_zn1", sys, t_ref)
        t, mn2, sd2, _ = mean_over_reps("ncoord_zn2", sys, t_ref)
        if t is None:
            continue
        if not np.all(np.isnan(mn1)):
            ax.fill_between(t, mn1 - sd1, mn1 + sd1, color=COLORS[sys], alpha=0.15)
            ax.plot(t, mn1, color=COLORS[sys], lw=1.0, ls="-",
                    label=f"{LABELS[sys]} Zn1")
        if not np.all(np.isnan(mn2)):
            ax.fill_between(t, mn2 - sd2, mn2 + sd2, color=COLORS[sys], alpha=0.15)
            ax.plot(t, mn2, color=COLORS[sys], lw=1.0, ls="--",
                    label=f"{LABELS[sys]} Zn2")
    ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Coordination number", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.set_xlim(0, TOTAL_NS)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_panel_label(ax, "B")

    # (c) ZnO cage bond distances for zno541, tern541 – violin+box
    ax = axes[2]
    vdata = {}
    for sys in ["zno541", "tern541"]:
        vals = production_values("zno_bond_mean", sys)
        if vals:
            vdata[sys] = vals
    if vdata:
        violin_box_panel(ax, vdata, "ZnO cage bond length (nm)")
    else:
        ax.text(0.5, 0.5, "ZnO bond data not available", transform=ax.transAxes,
                ha="center", color="0.5")
    add_panel_label(ax, "C", x=-0.12)
    ax.set_xlabel("System", fontsize=8)

    save_fig(fig, "fig10_zn_coordination")
    plt.close(fig)


# ─────────────────────────── Figure 11 – Energy ───────────────────────────

def _sanity_check_energy(arr: np.ndarray, name: str) -> bool:
    """
    Return True if the array looks physically sensible.
    Checks for plausible ranges to guard against GROMACS edr index mismatches.
    Ranges (approximate):
      conserved_kJ  : large negative, O(-1e6) kJ/mol
      potential_kJ  : moderate positive,  O(1e4) or large negative O(-1e6)
      kinetic_kJ    : positive, O(1e5)
      temperature_K : 250-350
      density_kgm3  : 900-1100
      pressure_bar  : order O(1000) or less
    """
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return False
    mn, std = float(a.mean()), float(a.std())
    rules = {
        "temperature_K": lambda m, s: 200 < m < 400 and s < 50,
        "density_kgm3":  lambda m, s: 800 < m < 1200 and s < 100,
        "pressure_bar":  lambda m, s: abs(m) < 2000 and s < 3000,
        "conserved_kJ":  lambda m, s: s < abs(m) * 0.1 and abs(m) > 1e3,
        "potential_kJ":  lambda m, s: abs(m) > 1e3,
        "kinetic_kJ":    lambda m, s: m > 0,
        "total_kJ":      lambda m, s: True,
    }
    check = rules.get(name)
    if check is None:
        return True
    return check(mn, std)


def _energy_series(ens: list, val_key: str, t_key: str | None = None,
                   n_atoms: int = 1) -> tuple:
    """
    Average a named energy quantity over replicas.
    Returns (t_comm, mean, std) or (None, None, None) if no valid data.
    """
    if t_key is None:
        t_key = f"t_ns_{val_key}"
    mn_list, t_list = [], []
    for e in ens:
        if val_key not in e:
            continue
        v = e[val_key] / n_atoms if n_atoms != 1 else e[val_key]
        if not _sanity_check_energy(e[val_key], val_key):
            continue
        tk = t_key if t_key in e else next(
            (k for k in e if k.startswith("t_ns_")), None)
        if tk is None:
            continue
        t_list.append(e[tk])
        mn_list.append(v)
    if not mn_list:
        return None, None, None
    min_len = min(len(v) for v in mn_list)
    t_comm  = t_list[0][:min_len]
    stk     = np.vstack([v[:min_len] for v in mn_list])
    return t_comm, stk.mean(0), stk.std(0, ddof=min(1, len(mn_list) - 1))


def figure11_energy():
    print("Figure 11: Conserved energy and density …")
    # Approximate solute atom counts (protein only; water is excluded from
    # timeseries analysis but NOT from conserved/potential energy in the edr).
    # Use as-is without per-atom normalisation to avoid unit confusion.
    N_ATOMS = {"apo": 10946, "naag": 10946 + 34, "zno541": 10946 + 48, "tern541": 10946 + 34 + 48}

    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 6.5 * MM_TO_IN * 25),
                             sharex=True, constrained_layout=True)

    any_conserved, any_density, any_pressure = False, False, False

    for sys in SYSTEMS:
        ens = [load_en(sys, rep) for rep in REPS]
        ens = [e for e in ens if e is not None]
        if not ens:
            continue
        na = N_ATOMS.get(sys, 10946)

        # Panel A – conserved energy (per solute atom)
        t, mn, sd = _energy_series(ens, "conserved_kJ", n_atoms=na)
        if t is not None:
            any_conserved = True
            axes[0].fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
            axes[0].plot(t, mn, color=COLORS[sys], lw=0.8, label=LABELS[sys])

        # Panel B – density
        t, mn, sd = _energy_series(ens, "density_kgm3")
        if t is not None:
            any_density = True
            axes[1].fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
            axes[1].plot(t, mn, color=COLORS[sys], lw=0.8, label=LABELS[sys])

        # Panel C – pressure
        t, mn, sd = _energy_series(ens, "pressure_bar")
        if t is not None:
            any_pressure = True
            axes[2].fill_between(t, mn - sd, mn + sd, color=COLORS[sys], alpha=0.2)
            axes[2].plot(t, mn, color=COLORS[sys], lw=0.8, label=LABELS[sys])

    for ax in axes:
        ax.axvline(SKIP_NS, color="gray", lw=0.7, ls="--")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.set_xlim(0, TOTAL_NS)

    axes[0].set_ylabel("Conserved energy\n(kJ mol⁻¹ atom⁻¹)", fontsize=8)
    axes[0].legend(fontsize=8, loc="upper right")
    if not any_conserved:
        axes[0].text(0.5, 0.5, "data not available / unphysical", transform=axes[0].transAxes,
                     ha="center", color="0.5")
    add_panel_label(axes[0], "A")

    axes[1].set_ylabel("Density (kg m⁻³)", fontsize=8)
    if not any_density:
        axes[1].text(0.5, 0.5, "data not available / unphysical", transform=axes[1].transAxes,
                     ha="center", color="0.5")
    add_panel_label(axes[1], "B")

    axes[2].set_xlabel("Time (ns)", fontsize=8)
    axes[2].set_ylabel("Pressure (bar)", fontsize=8)
    if not any_pressure:
        axes[2].text(0.5, 0.5, "data not available / unphysical", transform=axes[2].transAxes,
                     ha="center", color="0.5")
    add_panel_label(axes[2], "C")

    save_fig(fig, "fig11_energy")
    plt.close(fig)


# ─────────────────────────── Figure 12 – DCCM ─────────────────────────────

def figure12_dccm():
    print("Figure 12: DCCM heatmaps …")
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, FULL_W),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    cmap = mpl.colormaps.get_cmap("RdBu_r")

    for ax, sys in zip(axes_flat, SYSTEMS):
        mats = []
        for rep in REPS:
            d = load_ts(sys, rep)
            if d and "dccm_ca" in d:
                mats.append(d["dccm_ca"])
        if not mats:
            ax.text(0.5, 0.5, "no DCCM data", transform=ax.transAxes,
                    ha="center", color="0.5")
            ax.set_title(LABELS[sys], fontsize=8, color=COLORS[sys])
            continue

        dccm_mean = np.mean(mats, axis=0)
        # clamp symmetric
        vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(dccm_mean, cmap=cmap, norm=norm,
                       aspect="equal", interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cb.set_label("Cᵢⱼ", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(LABELS[sys], fontsize=8, color=COLORS[sys], fontweight="bold")
        ax.set_xlabel("Residue index", fontsize=7)
        ax.set_ylabel("Residue index", fontsize=7)
        ax.tick_params(labelsize=6)

    # panel labels
    for i, ax in enumerate(axes_flat):
        add_panel_label(ax, chr(65 + i))

    save_fig(fig, "fig12_dccm")
    plt.close(fig)


# ─────────────────────────── main ──────────────────────────────────────────

def main():
    print("=" * 60)
    print("PSMA-(ZnO)₁₂ figure generator")
    print(f"  Output directory: {FIG_DIR}")
    print("=" * 60)

    # quick data availability check
    print("\nData availability:")
    for sys in SYSTEMS:
        for rep in REPS:
            ts_ok = ts_path(sys, rep).exists()
            sm_ok = sm_path(sys, rep).exists()
            en_ok = en_path(sys, rep).exists()
            print(f"  {sys} rep{rep}:  "
                  f"timeseries={'OK' if ts_ok else 'MISSING'}  "
                  f"summary={'OK' if sm_ok else 'MISSING'}  "
                  f"energy={'OK' if en_ok else 'MISSING'}")
    print()

    generated = []

    def run(fn, label):
        try:
            fn()
            generated.append(label)
        except Exception as exc:
            print(f"  [ERROR] {label}: {exc}")
            import traceback
            traceback.print_exc()

    run(figure2_rmsd,           "Figure 2  – RMSD")
    run(figure3_rmsf,           "Figure 3  – RMSF")
    run(figure4_rg,             "Figure 4  – Radius of gyration")
    run(figure5_sasa,           "Figure 5  – Protein SASA")
    run(figure6_trp_sasa,       "Figure 6  – Trp SASA")
    run(figure7_hbonds,         "Figure 7  – H-bonds")
    run(figure8_distances,      "Figure 8  – Min-distance profiles")
    run(figure9_fret,           "Figure 9  – FRET analysis")
    run(figure10_zn_coordination, "Figure 10 – Zn coordination")
    run(figure11_energy,        "Figure 11 – Energy & density")
    run(figure12_dccm,          "Figure 12 – DCCM heatmaps")

    print()
    print("=" * 60)
    print(f"Successfully generated {len(generated)} figure(s):")
    for g in generated:
        print(f"  {g}")
    print(f"\nAll files saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
