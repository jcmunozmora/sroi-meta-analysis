#!/Users/jcmunoz/miniforge3/envs/ds/bin/python3
"""
factors_analysis_detailed.py
Detailed distribution and simulation analysis for each SROI adjustment factor.

Produces:
  - figures/fig_factor_deadweight.png
  - figures/fig_factor_attribution.png
  - figures/fig_factor_dropoff.png
  - figures/fig_factor_discount.png
  - figures/fig_factor_simulation.png   (multi-factor scenario)
  - figures/fig_factor_combined.png     (4-panel summary)
  - data/factor_benchmarks.json         (for web page)
  - data/factor_percentiles.csv         (for tables)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent
PROJECT_DIR = DATA_DIR.parent
FIG_DIR     = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {
    "deadweight":  "#2166ac",
    "attribution": "#1a9641",
    "dropoff":     "#d73027",
    "discount":    "#7b2d8b",
    "accent":      "#f4a582",
    "neutral":     "#d1d1d1",
}

SECTOR_ORDER = [
    "housing", "education", "employment", "health", "environment",
    "disability", "arts_culture", "social_inclusion", "youth",
    "agriculture_food", "community",
]

SECTOR_LABELS = {
    "housing": "Housing",
    "education": "Education",
    "employment": "Employment",
    "health": "Health",
    "environment": "Environment",
    "disability": "Disability",
    "arts_culture": "Arts & Culture",
    "social_inclusion": "Social Inclusion",
    "youth": "Youth",
    "agriculture_food": "Agriculture",
    "community": "Community",
}

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "sroi_clean_dataset_v3.csv")

# Clean adjustment factors (valid ranges)
def clean(col, lo, hi):
    s = df[col].copy()
    s[(s < lo) | (s > hi)] = np.nan
    return s

df["dw"]   = clean("deadweight_pct_lpdf",   0,   100)
df["at"]   = clean("attribution_pct_lpdf",  0,   100)
df["do"]   = clean("drop_off_pct_lpdf",     0,   100)
df["dr"]   = clean("discount_rate_pct_lpdf", 0.5, 20)
df["ratio"] = clean("sroi_ratio_lpdf",       0.1, 50)

# Only applied (>0 for adjustments)
df["dw_applied"]  = df["dw"].where(df["dw"] > 0)
df["at_applied"]  = df["at"].where(df["at"] > 0)
df["do_applied"]  = df["do"].where(df["do"] > 0)
df["dr_applied"]  = df["dr"]  # all discount rates > 0 by definition

# Forecast flag
df["is_forecast"] = df["report_type_clean"] == "Forecast"

# ── Helper functions ───────────────────────────────────────────────────────────

def percentile_table(series, name):
    s = series.dropna()
    return {
        "name": name,
        "n": int(len(s)),
        "mean": float(round(s.mean(), 2)),
        "median": float(round(s.median(), 2)),
        "sd": float(round(s.std(), 2)),
        "p10": float(round(s.quantile(0.10), 2)),
        "p25": float(round(s.quantile(0.25), 2)),
        "p75": float(round(s.quantile(0.75), 2)),
        "p90": float(round(s.quantile(0.90), 2)),
        "min": float(round(s.min(), 2)),
        "max": float(round(s.max(), 2)),
    }

def sector_breakdown(col, top_sectors=8):
    """Mean and n per sector, top sectors only."""
    result = []
    for sector in SECTOR_ORDER[:top_sectors]:
        mask = df["sector_clean"] == sector
        vals = df.loc[mask, col].dropna()
        vals = vals[vals > 0] if col != "dr_applied" else vals
        if len(vals) >= 3:
            result.append({
                "sector": SECTOR_LABELS.get(sector, sector.capitalize()),
                "n": len(vals),
                "mean": round(vals.mean(), 1),
                "median": round(vals.median(), 1),
            })
    return result

def country_breakdown(col, applied=True):
    result = {}
    for country in ["United Kingdom", "Australia", "United States", "Ireland"]:
        mask = df["country_clean"] == country
        vals = df.loc[mask, col].dropna()
        if applied:
            vals = vals[vals > 0]
        if len(vals) >= 5:
            result[country] = {
                "n": len(vals),
                "mean": round(vals.mean(), 1),
                "median": round(vals.median(), 1),
            }
    return result

def add_vline_annotation(ax, x, label, color="gray", ypos=0.85):
    ax.axvline(x, color=color, lw=1.5, ls="--", alpha=0.8)
    ax.text(x + 0.5, ax.get_ylim()[1] * ypos, label,
            color=color, fontsize=8, va="top", ha="left")

def kde_plot(ax, data, color, label, bw=0.3):
    s = data.dropna()
    if len(s) < 5:
        return
    s.hist(ax=ax, bins=20, density=True, alpha=0.35, color=color, edgecolor="white")
    kde = gaussian_kde(s, bw_method=bw)
    x_range = np.linspace(max(0, s.min() - 2), min(100, s.max() + 2), 300)
    ax.plot(x_range, kde(x_range), color=color, lw=2.5, label=label)
    return kde, x_range

def add_stats_box(ax, series, color, unit="%", x_frac=0.97, y_frac=0.97):
    s = series.dropna()
    text = (
        f"n = {len(s)}\n"
        f"Median: {s.median():.1f}{unit}\n"
        f"Mean: {s.mean():.1f}{unit}\n"
        f"SD: {s.std():.1f}{unit}\n"
        f"P10–P90: {s.quantile(0.1):.1f}–{s.quantile(0.9):.1f}{unit}"
    )
    ax.text(x_frac, y_frac, text,
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.85))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: DEADWEIGHT
# ══════════════════════════════════════════════════════════════════════════════

def plot_deadweight():
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # -- Panel A: Distribution (all applied)
    ax1 = fig.add_subplot(gs[0, :2])
    data = df["dw_applied"].dropna()
    kde_plot(ax1, data, PALETTE["deadweight"], "All reports with deadweight")
    ax1.axvline(data.median(), color=PALETTE["deadweight"], lw=2, ls="-", alpha=0.7)
    ax1.text(data.median() + 1, ax1.get_ylim()[1] * 0.92,
             f"Median\n{data.median():.0f}%",
             color=PALETTE["deadweight"], fontsize=9, fontweight="bold")
    ax1.axvspan(data.quantile(0.25), data.quantile(0.75),
                alpha=0.12, color=PALETTE["deadweight"], label="IQR")
    add_stats_box(ax1, data, PALETTE["deadweight"])
    ax1.set_xlabel("Deadweight rate (%)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("A. Distribution of Deadweight Rates\n(reports applying deadweight > 0%)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 100)

    # -- Panel B: Forecast vs Evaluative
    ax2 = fig.add_subplot(gs[0, 2])
    forecast_dw = df.loc[df["is_forecast"], "dw_applied"].dropna()
    eval_dw = df.loc[~df["is_forecast"], "dw_applied"].dropna()
    bp = ax2.boxplot([eval_dw, forecast_dw],
                     labels=["Evaluative\n(n={})".format(len(eval_dw)),
                             "Forecast\n(n={})".format(len(forecast_dw))],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#aec7e8")
    bp["boxes"][1].set_facecolor(PALETTE["deadweight"])
    ax2.set_ylabel("Deadweight rate (%)", fontsize=10)
    ax2.set_title("B. By Report Type", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    # Add medians as text
    for i, (vals, pos) in enumerate([(eval_dw, 1), (forecast_dw, 2)]):
        ax2.text(pos, vals.median() + 2, f"{vals.median():.0f}%",
                 ha="center", fontsize=9, fontweight="bold",
                 color=PALETTE["deadweight"] if i == 1 else "#4292c6")

    # -- Panel C: By sector (horizontal bars)
    ax3 = fig.add_subplot(gs[1, :2])
    sect_data = sector_breakdown("dw_applied")
    if sect_data:
        sectors = [d["sector"] for d in sect_data]
        medians = [d["median"] for d in sect_data]
        ns = [d["n"] for d in sect_data]
        colors_bar = [PALETTE["deadweight"] if m >= 20 else "#aec7e8" for m in medians]
        bars = ax3.barh(sectors, medians, color=colors_bar, edgecolor="white", height=0.65)
        ax3.axvline(data.median(), color="gray", lw=1.5, ls="--", label=f"Overall median ({data.median():.0f}%)")
        for bar, n in zip(bars, ns):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"n={n}", va="center", fontsize=8, color="gray")
        ax3.set_xlabel("Median deadweight rate (%)", fontsize=10)
        ax3.set_title("C. Median Deadweight by Sector", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.set_xlim(0, max(medians) * 1.3)

    # -- Panel D: Practitioner guidance
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    guidance = (
        "PRACTITIONER BENCHMARK\n"
        "─────────────────────\n"
        "Based on 154 SVI-database reports\n"
        "that applied deadweight > 0%\n\n"
        "Conservative:  10–15%\n"
        "Typical:       20–30%\n"
        "High:          35–50%\n\n"
        "Median:     20%\n"
        "Mean:       26%\n"
        "P10–P90:    5%–57%\n\n"
        "Common approaches:\n"
        "• National employment rate proxy\n"
        "• Survey of what participants\n"
        "  would have done otherwise\n"
        "• Comparison group / wait-list\n\n"
        "UK HM Treasury guidance:\n"
        "Apply where counterfactual\n"
        "evidence is available."
    )
    ax4.text(0.05, 0.95, guidance, transform=ax4.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#e6f0ff", ec=PALETTE["deadweight"], lw=1.5))

    fig.suptitle("Deadweight in SROI Practice\nDistribution from 383 SVI-Database Reports",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(FIG_DIR / "fig_factor_deadweight.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_deadweight.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_attribution():
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    data = df["at_applied"].dropna()
    kde_plot(ax1, data, PALETTE["attribution"], "All reports with attribution")
    ax1.axvline(data.median(), color=PALETTE["attribution"], lw=2, ls="-", alpha=0.7)
    ax1.text(data.median() + 1, ax1.get_ylim()[1] * 0.92,
             f"Median\n{data.median():.0f}%",
             color=PALETTE["attribution"], fontsize=9, fontweight="bold")
    ax1.axvspan(data.quantile(0.25), data.quantile(0.75),
                alpha=0.12, color=PALETTE["attribution"], label="IQR")
    add_stats_box(ax1, data, PALETTE["attribution"])
    ax1.set_xlabel("Attribution deduction (%)\n(share of outcomes attributed to other actors)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("A. Distribution of Attribution Deductions\n(reports applying attribution > 0%)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 100)

    ax2 = fig.add_subplot(gs[0, 2])
    forecast_at = df.loc[df["is_forecast"], "at_applied"].dropna()
    eval_at = df.loc[~df["is_forecast"], "at_applied"].dropna()
    bp = ax2.boxplot([eval_at, forecast_at],
                     labels=["Evaluative\n(n={})".format(len(eval_at)),
                             "Forecast\n(n={})".format(len(forecast_at))],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#a1d99b")
    bp["boxes"][1].set_facecolor(PALETTE["attribution"])
    ax2.set_ylabel("Attribution deduction (%)", fontsize=10)
    ax2.set_title("B. By Report Type", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    for i, (vals, pos) in enumerate([(eval_at, 1), (forecast_at, 2)]):
        ax2.text(pos, vals.median() + 2, f"{vals.median():.0f}%",
                 ha="center", fontsize=9, fontweight="bold",
                 color=PALETTE["attribution"] if i == 1 else "#41ab5d")

    ax3 = fig.add_subplot(gs[1, :2])
    sect_data = sector_breakdown("at_applied")
    if sect_data:
        sectors = [d["sector"] for d in sect_data]
        medians = [d["median"] for d in sect_data]
        ns = [d["n"] for d in sect_data]
        colors_bar = [PALETTE["attribution"] if m >= 25 else "#a1d99b" for m in medians]
        bars = ax3.barh(sectors, medians, color=colors_bar, edgecolor="white", height=0.65)
        ax3.axvline(data.median(), color="gray", lw=1.5, ls="--",
                    label=f"Overall median ({data.median():.0f}%)")
        for bar, n in zip(bars, ns):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"n={n}", va="center", fontsize=8, color="gray")
        ax3.set_xlabel("Median attribution deduction (%)", fontsize=10)
        ax3.set_title("C. Median Attribution by Sector", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.set_xlim(0, max(medians) * 1.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    guidance = (
        "PRACTITIONER BENCHMARK\n"
        "─────────────────────\n"
        "Based on 216 SVI-database reports\n"
        "that applied attribution > 0%\n\n"
        "Note: Attribution deduction =\n"
        "share attributed to OTHER actors.\n"
        "(100% – deduction = your share)\n\n"
        "Conservative:  10–20%\n"
        "Typical:       25–40%\n"
        "High:          40–60%\n\n"
        "Median:     25.5%\n"
        "Mean:       30.9%\n"
        "P10–P90:    5%–70%\n\n"
        "Common approaches:\n"
        "• Stakeholder survey ('what else\n"
        "  contributed to this change?')\n"
        "• Programme staff estimation\n"
        "• Multi-agency partnership share\n\n"
        "Apply per outcome, then weight."
    )
    ax4.text(0.05, 0.95, guidance, transform=ax4.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#e6ffe6", ec=PALETTE["attribution"], lw=1.5))

    fig.suptitle("Attribution in SROI Practice\nDistribution from 383 SVI-Database Reports",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(FIG_DIR / "fig_factor_attribution.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_attribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: DROP-OFF
# ══════════════════════════════════════════════════════════════════════════════

def plot_dropoff():
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    data = df["do_applied"].dropna()
    kde_plot(ax1, data, PALETTE["dropoff"], "All reports with drop-off")
    ax1.axvline(data.median(), color=PALETTE["dropoff"], lw=2, ls="-", alpha=0.7)
    ax1.text(data.median() + 1, ax1.get_ylim()[1] * 0.92,
             f"Median\n{data.median():.0f}%",
             color=PALETTE["dropoff"], fontsize=9, fontweight="bold")
    ax1.axvspan(data.quantile(0.25), data.quantile(0.75),
                alpha=0.12, color=PALETTE["dropoff"], label="IQR")
    add_stats_box(ax1, data, PALETTE["dropoff"])
    ax1.set_xlabel("Annual drop-off rate (%)\n(annual decline in outcomes after programme ends)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("A. Distribution of Annual Drop-Off Rates\n(reports applying drop-off > 0%)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 100)

    ax2 = fig.add_subplot(gs[0, 2])
    forecast_do = df.loc[df["is_forecast"], "do_applied"].dropna()
    eval_do = df.loc[~df["is_forecast"], "do_applied"].dropna()
    bp = ax2.boxplot([eval_do, forecast_do],
                     labels=["Evaluative\n(n={})".format(len(eval_do)),
                             "Forecast\n(n={})".format(len(forecast_do))],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#fc9272")
    bp["boxes"][1].set_facecolor(PALETTE["dropoff"])
    ax2.set_ylabel("Annual drop-off rate (%)", fontsize=10)
    ax2.set_title("B. By Report Type", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    for i, (vals, pos) in enumerate([(eval_do, 1), (forecast_do, 2)]):
        if len(vals) > 0:
            ax2.text(pos, vals.median() + 2, f"{vals.median():.0f}%",
                     ha="center", fontsize=9, fontweight="bold",
                     color=PALETTE["dropoff"] if i == 1 else "#cb181d")

    ax3 = fig.add_subplot(gs[1, :2])
    sect_data = sector_breakdown("do_applied")
    if sect_data:
        sectors = [d["sector"] for d in sect_data]
        medians = [d["median"] for d in sect_data]
        ns = [d["n"] for d in sect_data]
        colors_bar = [PALETTE["dropoff"] if m >= 20 else "#fc9272" for m in medians]
        bars = ax3.barh(sectors, medians, color=colors_bar, edgecolor="white", height=0.65)
        ax3.axvline(data.median(), color="gray", lw=1.5, ls="--",
                    label=f"Overall median ({data.median():.0f}%)")
        for bar, n in zip(bars, ns):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"n={n}", va="center", fontsize=8, color="gray")
        ax3.set_xlabel("Median annual drop-off rate (%)", fontsize=10)
        ax3.set_title("C. Median Drop-Off by Sector", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.set_xlim(0, max(medians) * 1.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    # Show cumulative outcome retention over time
    axins = ax4.inset_axes([0.05, 0.05, 0.9, 0.5])
    years = np.arange(0, 6)
    for label, rate, color in [
        ("Conservative (10%)", 0.10, "#aec7e8"),
        ("Typical (21.5%)",    0.215, PALETTE["dropoff"]),
        ("Aggressive (40%)",   0.40, "#67000d"),
    ]:
        retention = (1 - rate) ** years * 100
        axins.plot(years, retention, label=label, color=color, lw=2)
    axins.set_xlabel("Years after programme", fontsize=8)
    axins.set_ylabel("Outcome retained (%)", fontsize=8)
    axins.legend(fontsize=7, loc="upper right")
    axins.set_ylim(0, 110)
    axins.set_title("Cumulative retention", fontsize=8)
    axins.grid(alpha=0.3)

    guidance = (
        "PRACTITIONER BENCHMARK\n"
        "─────────────────────\n"
        "Based on 90 SVI-database reports\n"
        "that applied drop-off > 0%\n\n"
        "Conservative:   5–10%/yr\n"
        "Typical:        15–30%/yr\n"
        "High:           40–50%/yr\n\n"
        "Median:     21.5%/yr\n"
        "Mean:       29.1%/yr\n"
        "P10–P90:    5%–67%/yr\n\n"
        "Applied per year after year 1.\n"
        "At 20%/yr: 67% retained after 2 yrs"
    )
    ax4.text(0.05, 0.97, guidance, transform=ax4.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#fff0f0", ec=PALETTE["dropoff"], lw=1.5))

    fig.suptitle("Drop-Off in SROI Practice\nDistribution from 383 SVI-Database Reports",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(FIG_DIR / "fig_factor_dropoff.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_dropoff.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: DISCOUNT RATE
# ══════════════════════════════════════════════════════════════════════════════

def plot_discount():
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    data = df["dr_applied"].dropna()
    # Histogram with discrete bins around common rates
    ax1.hist(data, bins=np.arange(0.5, 16.5, 0.5), density=True,
             color=PALETTE["discount"], alpha=0.6, edgecolor="white")
    kde = gaussian_kde(data, bw_method=0.25)
    x = np.linspace(0.5, 16, 300)
    ax1.plot(x, kde(x), color=PALETTE["discount"], lw=2.5)
    # Mark key rates
    key_rates = [(3.5, "HM Treasury\n3.5%"), (3.0, "3.0%"), (5.0, "5.0%")]
    for rate, label in key_rates:
        ax1.axvline(rate, color="gray", lw=1, ls="--", alpha=0.7)
        ax1.text(rate + 0.08, ax1.get_ylim()[1] * 0.85, label,
                 color="gray", fontsize=8, va="top")
    add_stats_box(ax1, data, PALETTE["discount"], unit="%")
    ax1.set_xlabel("Discount rate (%)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("A. Distribution of Discount Rates\n(n=246 reports with documented discount rate)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 16)

    # Value of £1 at different discount rates over time
    ax2 = fig.add_subplot(gs[0, 2])
    years = np.arange(0, 11)
    for label, rate, color in [
        ("2.5%",  0.025, "#c994c7"),
        ("3.5%\n(HM Treasury)", 0.035, PALETTE["discount"]),
        ("5.0%",  0.050, "#dd1c77"),
        ("10.0%", 0.100, "#67000d"),
    ]:
        pv = (1 / (1 + rate)) ** years
        ax2.plot(years, pv, label=label, lw=2, color=color)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Present value of £1", fontsize=10)
    ax2.set_title("B. Discounting Effect\nOver Time", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, title="Discount rate")
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(alpha=0.3)

    # By country
    ax3 = fig.add_subplot(gs[1, :2])
    countries = []
    country_medians = []
    country_ns = []
    for country in ["United Kingdom", "Australia", "United States", "Ireland", "Unknown"]:
        mask = df["country_clean"] == country
        vals = df.loc[mask, "dr_applied"].dropna()
        if len(vals) >= 5:
            countries.append(country)
            country_medians.append(vals.median())
            country_ns.append(len(vals))

    colors_bar = [PALETTE["discount"] if m == 3.5 else "#c994c7" for m in country_medians]
    bars = ax3.barh(countries, country_medians, color=colors_bar, edgecolor="white", height=0.55)
    ax3.axvline(3.5, color="gray", lw=1.5, ls="--", label="HM Treasury rate (3.5%)")
    for bar, n in zip(bars, country_ns):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"n={n}", va="center", fontsize=9, color="gray")
    ax3.set_xlabel("Median discount rate (%)", fontsize=10)
    ax3.set_title("C. Discount Rate by Country", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, max(country_medians) * 1.4 if country_medians else 10)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    guidance = (
        "PRACTITIONER BENCHMARK\n"
        "─────────────────────\n"
        "Based on 246 SVI-database reports\n"
        "with documented discount rate\n\n"
        "UK (HM Treasury):    3.5%\n"
        "Australia (ABS):     3.5–5.0%\n"
        "Other / not stated:  3.0–5.0%\n\n"
        "Median:      3.5%\n"
        "Mean:        4.1%\n"
        "P10–P90:     3.0%–5.0%\n\n"
        "Recommendation:\n"
        "Use 3.5% (UK HM Treasury) for\n"
        "UK-based studies. Report the\n"
        "rate used and cite the source.\n\n"
        "Sensitivity: test at 1.5% and 5%\n"
        "to show robustness."
    )
    ax4.text(0.05, 0.95, guidance, transform=ax4.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f5e6ff", ec=PALETTE["discount"], lw=1.5))

    fig.suptitle("Discount Rate in SROI Practice\nDistribution from 383 SVI-Database Reports",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(FIG_DIR / "fig_factor_discount.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_discount.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: 4-PANEL COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle("SROI Adjustment Factor Distributions\n383 Reports — Social Value International Database",
    #              fontsize=14, fontweight="bold", y=1.01)

    configs = [
        (axes[0, 0], df["dw_applied"],  PALETTE["deadweight"],  "Deadweight (%)",    "A. Deadweight (n={})"),
        (axes[0, 1], df["at_applied"],  PALETTE["attribution"], "Attribution deduction (%)", "B. Attribution (n={})"),
        (axes[1, 0], df["do_applied"],  PALETTE["dropoff"],     "Annual drop-off (%)", "C. Drop-Off (n={})"),
        (axes[1, 1], df["dr_applied"],  PALETTE["discount"],    "Discount rate (%)",  "D. Discount Rate (n={})"),
    ]

    for ax, col, color, xlabel, title_tpl in configs:
        data = col.dropna()
        n = len(data)
        data.hist(ax=ax, bins=20, density=True, alpha=0.4, color=color, edgecolor="white")
        try:
            kde = gaussian_kde(data, bw_method=0.35)
            x = np.linspace(max(0, data.min() - 1), min(100, data.max() + 1), 300)
            ax.plot(x, kde(x), color=color, lw=2.5)
        except Exception:
            pass
        # Median line
        med = data.median()
        ax.axvline(med, color=color, lw=2, ls="--", alpha=0.9)
        ax.text(med + 0.5, ax.get_ylim()[1] * 0.85,
                f"Median: {med:.1f}%", color=color, fontsize=9, fontweight="bold")
        # IQR shading
        ax.axvspan(data.quantile(0.25), data.quantile(0.75),
                   alpha=0.12, color=color)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(title_tpl.format(n), fontsize=11, fontweight="bold")
        # Stats box
        text = f"Mean: {data.mean():.1f}%\nMedian: {med:.1f}%\nSD: {data.std():.1f}%"
        ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_factor_combined.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_combined.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: RATIO SIMULATION — How factors affect the SROI ratio
# ══════════════════════════════════════════════════════════════════════════════

def plot_simulation():
    """Show how the SROI ratio changes under different adjustment combinations."""
    np.random.seed(42)
    N_SIM = 10000
    base_ratio = 5.0  # hypothetical unadjusted ratio

    # Scenarios: (label, dw, at, do, dr, color)
    scenarios = [
        ("No adjustments\n(as often reported)", 0, 0, 0, 0, "#d73027"),
        ("Deadweight only\n(DW=20%)", 0.20, 0, 0, 0, "#fc8d59"),
        ("DW + Attribution\n(DW=20%, AT=25%)", 0.20, 0.25, 0, 0, "#fee090"),
        ("DW + AT + Drop-off\n(DW=20%, AT=25%, DO=20%/yr)", 0.20, 0.25, 0.20, 0, "#e0f3f8"),
        ("All four adjustments\n(DW=20%, AT=25%,\nDO=20%, DR=3.5%)", 0.20, 0.25, 0.20, 0.035, "#4575b4"),
        ("Fully calibrated\n(from observed data)", None, None, None, None, "#313695"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    # fig.suptitle("How Adjustment Factors Affect the SROI Ratio\n"
    #              "Simulation based on 383-report empirical distributions",
    #              fontsize=13, fontweight="bold")

    corrected_ratios = []
    labels = []

    for label, dw, at, do_, dr, color in scenarios:
        if label.startswith("Fully"):
            # Draw from observed distributions
            dw_s = np.random.choice(df["dw_applied"].dropna().values, N_SIM, replace=True)
            at_s = np.random.choice(df["at_applied"].dropna().values, N_SIM, replace=True)
            do_s = np.random.choice(df["do_applied"].dropna().values, N_SIM, replace=True) / 100
            dr_s = np.random.choice(df["dr_applied"].dropna().values, N_SIM, replace=True) / 100
            # Apply corrections: ratio * (1-dw) * (1-at) * (1-do)^2 / discount
            corrected = base_ratio * (1 - dw_s / 100) * (1 - at_s / 100) * (1 - do_s) ** 2
        else:
            dw_frac = dw or 0
            at_frac = at or 0
            do_frac = do_ or 0
            dr_frac = dr or 0
            # Simple scalar correction
            corrected_val = base_ratio * (1 - dw_frac) * (1 - at_frac) * (1 - do_frac) ** 2
            corrected = np.full(N_SIM, corrected_val)

        corrected_ratios.append(corrected)
        labels.append(label)

    # Panel A: Horizontal boxplot
    bp = ax1.boxplot(corrected_ratios[::-1],
                     vert=False,
                     labels=labels[::-1],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2),
                     flierprops=dict(marker=".", markersize=2, alpha=0.3))
    colors_rev = [s[5] for s in scenarios][::-1]
    for patch, color in zip(bp["boxes"], colors_rev):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.axvline(base_ratio, color="red", lw=1.5, ls="--", alpha=0.8, label=f"Unadjusted ratio ({base_ratio}:1)")
    ax1.set_xlabel("Corrected SROI ratio", fontsize=10)
    ax1.set_title("A. Ratio Correction Under Different\nAdjustment Combinations", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, base_ratio * 1.3)

    # Panel B: Tornado chart — sensitivity to each factor
    ax2.axis("off")
    # Show percentage reduction from each factor
    dw_effect  = base_ratio * 0.20   # 20% deadweight
    at_effect  = base_ratio * (1 - 0.20) * 0.25  # 25% attribution after deadweight
    do_effect  = base_ratio * (1 - 0.20) * (1 - 0.25) * (1 - (1 - 0.20)**2)
    remaining  = base_ratio * (1 - 0.20) * (1 - 0.25) * (1 - 0.20) ** 2

    factors_tornado = [
        ("Unadjusted ratio",              base_ratio, "#d73027"),
        ("After deadweight (20%)",        base_ratio * 0.80, "#fc8d59"),
        ("After attribution (25%)",       base_ratio * 0.80 * 0.75, "#fee090"),
        ("After drop-off (20%/yr × 2yr)", base_ratio * 0.80 * 0.75 * 0.64, "#91bfdb"),
        ("After all four factors",        remaining, "#4575b4"),
    ]

    y_pos = range(len(factors_tornado))
    colors_t = [f[2] for f in factors_tornado]
    values_t = [f[1] for f in factors_tornado]
    lbls_t = [f[0] for f in factors_tornado]

    bars_t = ax2.barh(list(y_pos), values_t, color=colors_t, edgecolor="white", height=0.6,
                      transform=ax2.get_xaxis_transform())
    ax2 = fig.add_subplot(1, 2, 2)  # re-add properly
    for i, (label, val, color) in enumerate(factors_tornado):
        ax2.barh(i, val, color=color, edgecolor="white", height=0.6)
        ax2.text(val + 0.05, i, f"{val:.2f}:1", va="center", fontsize=9, fontweight="bold")

    ax2.set_yticks(range(len(factors_tornado)))
    ax2.set_yticklabels(lbls_t, fontsize=9)
    ax2.set_xlabel("SROI ratio", fontsize=10)
    ax2.set_title("B. Sequential Adjustment Effect\n(Base ratio = 5:1, typical adjustments)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(0, base_ratio * 1.35)
    ax2.axvline(base_ratio, color="red", lw=1.5, ls="--", alpha=0.6, label="Unadjusted")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    reduction_pct = (1 - remaining / base_ratio) * 100
    ax2.text(0.97, 0.03,
             f"Total reduction: {reduction_pct:.0f}%\n"
             f"({base_ratio:.1f}:1 → {remaining:.2f}:1)",
             transform=ax2.transAxes, fontsize=9,
             va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#4575b4", alpha=0.9))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_factor_simulation.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig_factor_simulation.png")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT BENCHMARK DATA (JSON for web page)
# ══════════════════════════════════════════════════════════════════════════════

def export_benchmarks():
    benchmarks = {
        "metadata": {
            "n_reports": 383,
            "source": "Social Value International reports database",
            "extracted": "2026-03",
            "method": "Full-PDF LLM extraction (GPT-4o-mini + GPT-4o reconciler)",
        },
        "factors": {
            "deadweight": percentile_table(df["dw_applied"], "Deadweight rate (%)"),
            "attribution": percentile_table(df["at_applied"], "Attribution deduction (%)"),
            "drop_off": percentile_table(df["do_applied"], "Annual drop-off rate (%)"),
            "discount_rate": percentile_table(df["dr_applied"], "Discount rate (%)"),
            "sroi_ratio": percentile_table(df["ratio"], "SROI ratio"),
        },
        "sector_breakdown": {
            "deadweight": sector_breakdown("dw_applied"),
            "attribution": sector_breakdown("at_applied"),
            "drop_off": sector_breakdown("do_applied"),
        },
        "country_breakdown": {
            "discount_rate": country_breakdown("dr_applied", applied=False),
            "deadweight": country_breakdown("dw_applied"),
        },
        "by_report_type": {
            "deadweight": {
                "forecast": percentile_table(df.loc[df["is_forecast"], "dw_applied"], "Deadweight - Forecast"),
                "evaluative": percentile_table(df.loc[~df["is_forecast"], "dw_applied"], "Deadweight - Evaluative"),
            },
            "attribution": {
                "forecast": percentile_table(df.loc[df["is_forecast"], "at_applied"], "Attribution - Forecast"),
                "evaluative": percentile_table(df.loc[~df["is_forecast"], "at_applied"], "Attribution - Evaluative"),
            },
        },
    }

    out_path = DATA_DIR / "factor_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(benchmarks, f, indent=2, default=str)
    print(f"  ✓ factor_benchmarks.json ({out_path.stat().st_size//1024} KB)")

    # Also export percentile CSV for tables
    rows = []
    for fname, col in [("Deadweight", "dw_applied"), ("Attribution", "at_applied"),
                        ("Drop-off", "do_applied"), ("Discount rate", "dr_applied")]:
        s = df[col].dropna()
        rows.append({
            "Factor": fname, "n": len(s),
            "P10": round(s.quantile(0.10), 1),
            "P25": round(s.quantile(0.25), 1),
            "Median": round(s.median(), 1),
            "Mean": round(s.mean(), 1),
            "P75": round(s.quantile(0.75), 1),
            "P90": round(s.quantile(0.90), 1),
            "SD": round(s.std(), 1),
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "factor_percentiles.csv", index=False)
    print("  ✓ factor_percentiles.csv")


# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating factor distribution figures...")
    plot_deadweight()
    plot_attribution()
    plot_dropoff()
    plot_discount()
    plot_combined()
    plot_simulation()
    export_benchmarks()
    print("\nAll done!")
