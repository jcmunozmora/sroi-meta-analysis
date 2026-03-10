#!/usr/bin/env python3
"""
analysis.py — Statistical analysis for SROI meta-analysis paper
Produces: figures + tables for draft_v1.qmd

Outputs (all in ../figures/):
  fig1_sector_distribution.pdf
  fig2_quality_by_principle.pdf
  fig3_quality_distribution.pdf
  fig4_quality_by_sector.pdf
  fig5_quality_assured_vs_not.pdf
  fig6_stakeholders_vs_quality.pdf
  fig7_sroi_ratio_distribution.pdf
  regression_table.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data" / "sroi_clean_dataset.csv"
FIGDIR  = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
BLUE  = "#2C5F8A"
ORANGE= "#E07B39"
GREY  = "#7A8694"
GREEN = "#3A7D44"
RED   = "#C0392B"
LBLUE = "#A8C5DA"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Dataset: {len(df)} reports, {df.columns.tolist()}")

PRINCIPLES = [
    "p1_involve_stakeholders",
    "p2_understand_changes",
    "p3_value_what_matters",
    "p4_only_material",
    "p5_do_not_overclaim",
    "p6_be_transparent",
    "p7_verify_result",
    "p8_be_responsive",
]
P_LABELS = [
    "P1: Involve\nstakeholders",
    "P2: Understand\nchanges",
    "P3: Value\nwhat matters",
    "P4: Only include\nmaterial",
    "P5: Do not\nover-claim",
    "P6: Be\ntransparent",
    "P7: Verify\nthe result",
    "P8: Be\nresponsive",
]

# =============================================================================
# FIGURE 1 — Sector distribution (horizontal bar)
# =============================================================================
sect_counts = df["sector_clean"].value_counts()
sect_counts = sect_counts[sect_counts >= 2].sort_values()
sect_labels = {
    "housing": "Housing", "education": "Education", "employment": "Employment",
    "health": "Health", "environment": "Environment", "disability": "Disability",
    "arts_culture": "Arts & Culture", "other": "Other", "agriculture_food": "Agriculture/Food",
    "social_inclusion": "Social Inclusion", "youth": "Youth", "community": "Community",
    "sports": "Sports", "justice": "Justice", "elderly": "Elderly",
}

fig, ax = plt.subplots(figsize=(7, 5))
ys  = range(len(sect_counts))
xs  = sect_counts.values
lbls = [sect_labels.get(s, s.title()) for s in sect_counts.index]
bars = ax.barh(ys, xs, color=BLUE, alpha=0.85, height=0.65)
ax.set_yticks(ys)
ax.set_yticklabels(lbls, fontsize=9)
ax.set_xlabel("Number of reports")
# ax.set_title("Figure 1. Sectoral distribution of SROI reports\n(Social Value UK database, n=383)",
#              fontsize=10, loc="left", pad=8)
for bar, val in zip(bars, xs):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8, color=GREY)
ax.set_xlim(0, xs.max() * 1.15)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig(FIGDIR / "fig1_sector_distribution.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig1_sector_distribution.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 1 saved")

# =============================================================================
# FIGURE 2 — Quality scores by principle (stacked bar: 0, 1, 2)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
n = df[PRINCIPLES[0]].count()
x = np.arange(len(PRINCIPLES))
w = 0.6

for i, p in enumerate(PRINCIPLES):
    total = df[p].count()
    v0 = (df[p] == 0).sum() / total * 100
    v1 = (df[p] == 1).sum() / total * 100
    v2 = (df[p] == 2).sum() / total * 100
    ax.bar(i, v0, w, color=RED,   alpha=0.85, label="Score 0 (absent)"    if i == 0 else "")
    ax.bar(i, v1, w, bottom=v0,    color=ORANGE, alpha=0.85, label="Score 1 (partial)" if i == 0 else "")
    ax.bar(i, v2, w, bottom=v0+v1, color=GREEN,  alpha=0.85, label="Score 2 (full)"    if i == 0 else "")
    # Mean score annotation
    mean_val = df[p].mean()
    ax.text(i, 103, f"{mean_val:.2f}", ha="center", fontsize=8, color="black", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(P_LABELS, fontsize=8.5)
ax.set_ylabel("Percentage of reports (%)")
ax.set_ylim(0, 115)
# ax.set_title("Figure 2. Compliance with SVI's eight reporting principles\n(% of reports by score, mean score shown above bars; max=2.0)",
#              fontsize=10, loc="left", pad=8)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.axhline(50, color=GREY, lw=0.8, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGDIR / "fig2_quality_by_principle.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig2_quality_by_principle.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 2 saved")

# =============================================================================
# FIGURE 3 — Distribution of quality scores (histogram)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# Left: histogram of quality_pct
ax = axes[0]
qdata = df["quality_pct"].dropna()
ax.hist(qdata, bins=20, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(qdata.mean(),   color=RED,    lw=2, linestyle="-",  label=f"Mean = {qdata.mean():.1f}%")
ax.axvline(qdata.median(), color=ORANGE, lw=2, linestyle="--", label=f"Median = {qdata.median():.1f}%")
ax.set_xlabel("Overall quality score (%)")
ax.set_ylabel("Number of reports")
ax.set_title("(a) Distribution of overall quality scores", fontsize=10)
ax.legend(fontsize=9)

# Right: box plots by assurance status
ax = axes[1]
assured_q     = df[df.assurance_clean == 1]["quality_pct"].dropna()
not_assured_q = df[df.assurance_clean == 0]["quality_pct"].dropna()
bp = ax.boxplot(
    [not_assured_q, assured_q],
    labels=["Not assured\n(n=320)", "Assured\n(n=63)"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=GREY),
    capprops=dict(color=GREY),
    flierprops=dict(marker="o", markerfacecolor=GREY, markersize=3, alpha=0.5),
)
bp["boxes"][0].set_facecolor(LBLUE)
bp["boxes"][1].set_facecolor(BLUE)
t_stat, p_val = stats.ttest_ind(assured_q, not_assured_q)
cohen_d = (assured_q.mean() - not_assured_q.mean()) / df["quality_pct"].std()
ax.set_ylabel("Quality score (%)")
ax.set_title(f"(b) Quality by assurance status\nt={t_stat:.2f}, p<0.001, Cohen's d={cohen_d:.2f}", fontsize=10)
ax.axhline(qdata.mean(), color=GREY, lw=0.8, linestyle="--", alpha=0.5)

# fig.suptitle("Figure 3. Distribution of overall quality scores across 376 SROI reports",
#              fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(FIGDIR / "fig3_quality_distribution.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig3_quality_distribution.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 3 saved")

# =============================================================================
# FIGURE 4 — Quality by sector (dot plot with CI)
# =============================================================================
sect_stats = (
    df[df.sector_clean.notna()]
    .groupby("sector_clean")["quality_pct"]
    .agg(["mean", "std", "count"])
    .query("count >= 5")
    .sort_values("mean")
)
sect_stats["se"]  = sect_stats["std"] / np.sqrt(sect_stats["count"])
sect_stats["ci95"] = 1.96 * sect_stats["se"]
overall_mean = df["quality_pct"].mean()

fig, ax = plt.subplots(figsize=(7, 5))
ys = range(len(sect_stats))
ax.axvline(overall_mean, color=GREY, lw=1.2, linestyle="--", alpha=0.7,
           label=f"Overall mean ({overall_mean:.1f}%)")
ax.barh(ys, sect_stats["mean"], xerr=sect_stats["ci95"],
        color=[GREEN if m > overall_mean else LBLUE for m in sect_stats["mean"]],
        height=0.6, capsize=3, error_kw=dict(elinewidth=1.2, capthick=1.2, ecolor=GREY),
        alpha=0.85)
ax.set_yticks(ys)
ax.set_yticklabels([sect_labels.get(s, s.title()) + f" (n={int(c)})"
                    for s, c in zip(sect_stats.index, sect_stats["count"])], fontsize=9)
for i, (mean, ci) in enumerate(zip(sect_stats["mean"], sect_stats["ci95"])):
    ax.text(mean + ci + 0.5, i, f"{mean:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Mean quality score (%) ± 95% CI")
# ax.set_title("Figure 4. Mean quality scores by sector\n(bars: 95% CI; green = above overall mean)",
#              fontsize=10, loc="left", pad=8)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 80)
plt.tight_layout()
plt.savefig(FIGDIR / "fig4_quality_by_sector.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig4_quality_by_sector.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 4 saved")

# =============================================================================
# FIGURE 5 — Stakeholder mentions vs quality score (scatter + regression)
# =============================================================================
sub = df[["stakeholder_mentions", "quality_pct", "assurance_clean"]].dropna()
r_sp, p_sp = stats.spearmanr(sub["stakeholder_mentions"], sub["quality_pct"])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
assured_mask = sub["assurance_clean"] == 1
ax.scatter(sub.loc[~assured_mask, "stakeholder_mentions"],
           sub.loc[~assured_mask, "quality_pct"],
           alpha=0.35, s=18, color=LBLUE, label="Not assured")
ax.scatter(sub.loc[assured_mask, "stakeholder_mentions"],
           sub.loc[assured_mask, "quality_pct"],
           alpha=0.7, s=30, color=BLUE, label="Assured", zorder=5)
# Log-linear fit line
x_fit = np.linspace(0, sub["stakeholder_mentions"].max(), 200)
log_x = np.log1p(sub["stakeholder_mentions"])
slope, intercept, r_val, p_val_ols, _ = stats.linregress(log_x, sub["quality_pct"])
ax.plot(x_fit, intercept + slope * np.log1p(x_fit), color=RED, lw=2,
        label=f"Log-linear fit (r={r_val:.2f})")
ax.set_xlabel("Stakeholder keyword mentions in PDF text")
ax.set_ylabel("Overall quality score (%)")
# ax.set_title(
#     f"Figure 5. Stakeholder engagement and reporting quality\n"
#     f"Spearman ρ = {r_sp:.3f}, p < 0.001 (n={len(sub)})",
#     fontsize=10, loc="left", pad=8)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGDIR / "fig5_stakeholders_vs_quality.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig5_stakeholders_vs_quality.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 5 saved")

# =============================================================================
# FIGURE 6 — SROI ratio distribution (log scale histogram)
# =============================================================================
ratios = df["sroi_ratio_value"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

ax = axes[0]
ax.hist(ratios, bins=25, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(ratios.mean(),   color=RED,    lw=2, label=f"Mean = {ratios.mean():.1f}:1")
ax.axvline(ratios.median(), color=ORANGE, lw=2, linestyle="--", label=f"Median = {ratios.median():.1f}:1")
ax.set_xlabel("SROI ratio")
ax.set_ylabel("Number of reports")
ax.set_title("(a) Linear scale", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(np.log(ratios), bins=25, color=ORANGE, alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axvline(np.log(ratios.mean()),   color=RED,    lw=2, label=f"log(Mean)")
ax.axvline(np.log(ratios.median()), color=BLUE,   lw=2, linestyle="--", label=f"log(Median)")
ax.set_xlabel("log(SROI ratio)")
ax.set_ylabel("Number of reports")
ax.set_title("(b) Log scale — closer to normal", fontsize=10)
ax.legend(fontsize=9)
w_r, p_r = stats.shapiro(np.log(ratios))
ax.text(0.97, 0.95, f"Shapiro-Wilk (log): W={w_r:.3f}, p={p_r:.3f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="top", color=GREY)

# fig.suptitle(f"Figure 6. Distribution of SROI ratios (n={len(ratios)} reports with extractable ratios)",
#              fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(FIGDIR / "fig6_sroi_ratio_distribution.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig6_sroi_ratio_distribution.png", bbox_inches="tight", dpi=200)
plt.close()
print("✓ Figure 6 saved")

# =============================================================================
# REGRESSION TABLE
# =============================================================================
reg = df.copy()
reg = reg[reg.country_clean.notna() & (reg.country_clean != "Unknown")]
reg = reg[reg.org_type.notna() & (reg.org_type != "unknown")]
reg = reg[reg.sector_clean.notna()]

top_sectors = ["education", "employment", "health", "environment", "disability", "arts_culture"]
for s in top_sectors:
    reg[f"sec_{s}"] = (reg.sector_clean == s).astype(float)

for c in ["Australia", "USA", "Canada"]:
    reg[f"cty_{c}"] = (reg.country_clean == c).astype(float)

for o in ["government", "private", "social_enterprise"]:
    reg[f"org_{o}"] = (reg.org_type == o).astype(float)

reg["assured"]          = reg["assurance_clean"].astype(float)
yr_mean = reg["year_clean"].mean()
yr_std  = reg["year_clean"].std()
reg["year_std"]         = (reg["year_clean"] - yr_mean) / yr_std
reg["log_stakeholders"] = np.log1p(reg["stakeholder_mentions"])

feature_cols = (
    [f"sec_{s}" for s in top_sectors] +
    [f"cty_{c}" for c in ["Australia", "USA", "Canada"]] +
    [f"org_{o}" for o in ["government", "private", "social_enterprise"]] +
    ["assured", "year_std", "log_stakeholders"]
)
reg_clean = reg[feature_cols + ["quality_pct"]].dropna()
X = sm.add_constant(reg_clean[feature_cols])
y = reg_clean["quality_pct"]
model = sm.OLS(y, X).fit(cov_type="HC3")

# Export regression table
reg_tbl = model.summary2().tables[1].copy()
reg_tbl.columns = ["Coef.", "Std. Err.", "z", "P>|z|", "CI_low", "CI_high"]
reg_tbl["Sig."] = reg_tbl["P>|z|"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
)
reg_tbl.to_csv(BASE / "data" / "regression_table.csv")
print(f"\n✓ Regression table saved")
print(f"  N={len(reg_clean)}, R²={model.rsquared:.3f}, Adj. R²={model.rsquared_adj:.3f}")
print(f"  F={model.fvalue:.2f}, p(F)={model.f_pvalue:.4f}")

# =============================================================================
# PRINT KEY STATISTICS FOR PAPER
# =============================================================================
print("\n" + "="*60)
print("KEY STATISTICS FOR PAPER")
print("="*60)

# Normality
w_q, p_q = stats.shapiro(df["quality_pct"].dropna())
print(f"\nShapiro-Wilk (quality_pct): W={w_q:.4f}, p={p_q:.4f} → {'NOT normal' if p_q<0.05 else 'normal'}")

# Assurance t-test
t_s, p_t = stats.ttest_ind(
    df[df.assurance_clean==1]["quality_pct"].dropna(),
    df[df.assurance_clean==0]["quality_pct"].dropna()
)
d_cohen = (df[df.assurance_clean==1]["quality_pct"].mean() -
           df[df.assurance_clean==0]["quality_pct"].mean()) / df["quality_pct"].std()
print(f"\nAssurance t-test: t={t_s:.3f}, p={p_t:.6f}, Cohen's d={d_cohen:.3f}")

# Kruskal-Wallis by sector
sect_groups = [df[df.sector_clean==s]["quality_pct"].dropna().values
               for s in df.groupby("sector_clean")["quality_pct"].count()[lambda x: x>=10].index]
h_kw, p_kw = stats.kruskal(*sect_groups)
print(f"\nKruskal-Wallis (sector, n>=10): H={h_kw:.3f}, p={p_kw:.4f}")

# Spearman correlations
r_st, p_st = stats.spearmanr(df["stakeholder_mentions"].dropna(),
                              df.loc[df["stakeholder_mentions"].notna(), "quality_pct"])
print(f"\nSpearman (stakeholders vs quality): ρ={r_st:.3f}, p={p_st:.4f}")

r_yr, p_yr = stats.spearmanr(df["year_clean"].dropna(),
                              df.loc[df["year_clean"].notna(), "quality_pct"])
print(f"Spearman (year vs quality): ρ={r_yr:.3f}, p={p_yr:.4f}")

# Key regression coefficients
print(f"\nOLS regression (HC3 robust SE):")
print(f"  N={len(reg_clean)}, R²={model.rsquared:.3f}, Adj R²={model.rsquared_adj:.3f}")
key_vars = ["assured", "log_stakeholders", "sec_health", "cty_Australia"]
for v in key_vars:
    if v in reg_tbl.index:
        row = reg_tbl.loc[v]
        print(f"  {v}: β={row['Coef.']:.3f}, SE={row['Std. Err.']:.3f}, p={row['P>|z|']:.4f} {row['Sig.']}")

print("\n✓ All figures and tables complete")
print(f"  Figures saved to: {FIGDIR}")
