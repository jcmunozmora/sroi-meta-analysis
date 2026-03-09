#!/usr/bin/env python3
"""
SROI Calculation Factors Analysis — generates figures 13-16 for the
"Anatomy of SROI Calculation Elements" section.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = '/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT/Papers/2026_sroi'
OUT  = f'{BASE}/figures'

COLORS = {
    'primary':    '#2E4057',
    'secondary':  '#048A81',
    'accent':     '#54C6EB',
    'warning':    '#EF8C4C',
    'danger':     '#C84B31',
    'light':      '#E8F4F8',
    'assured':    '#2E8B57',
    'not_assured':'#C84B31',
}

# ─── Load data ────────────────────────────────────────────────────────────────
df       = pd.read_csv(f'{BASE}/data/sroi_clean_dataset.csv')
factors  = pd.read_csv(f'{BASE}/data/sroi_factors.csv')
m        = df.merge(factors, on='id', how='left')

# Derived columns
m['all_four'] = ((m['deadweight']==1)&(m['attribution']==1)&
                 (m['drop_off']==1)&(m['displacement']==1)).astype(int)
m['any_one']  = ((m['deadweight']==1)|(m['attribution']==1)|
                 (m['drop_off']==1)|(m['displacement']==1)).astype(int)
m['dw_and_at']= ((m['deadweight']==1)&(m['attribution']==1)).astype(int)
m['n_factors'] = m[['deadweight','attribution','drop_off','displacement']].sum(axis=1)

N = len(m)

# ─── Figure 13: SROI Calculation Cascade ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

stages = [
    ("Stakeholder\nengagement",   m['p1_involve_stakeholders'].gt(0).mean()),
    ("Theory of\nchange",         m['theory_of_change'].mean()),
    ("Financial\nproxy/value",    m['proxy'].mean()),
    ("Deadweight",                m['deadweight'].mean()),
    ("Attribution",               m['attribution'].mean()),
    ("Drop-off",                  m['drop_off'].mean()),
    ("Displacement",              m['displacement'].mean()),
    ("All four\nadjustments",     m['all_four'].mean()),
    ("Sensitivity\nanalysis",     m['p7_verify_result'].gt(0).mean()),
]

labels = [s[0] for s in stages]
vals   = [s[1]*100 for s in stages]

# Color by type
bar_colors = [COLORS['secondary']] * 3 + \
             [COLORS['danger']] * 4 + \
             [COLORS['warning']] + \
             [COLORS['primary']]

bars = ax.bar(range(len(stages)), vals, color=bar_colors, alpha=0.88,
              edgecolor='white', linewidth=0.8, width=0.7)

# Annotations
for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#333333')

ax.set_xticks(range(len(stages)))
ax.set_xticklabels(labels, fontsize=9, ha='center')
ax.set_ylabel('% of reports with evidence of engagement', fontsize=11)
ax.set_title('Figure 13: SROI Calculation Elements — Cascade of Compliance\n'
             'across the six-stage SROI process (n=383 reports)',
             fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, max(vals)*1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=50, color='grey', linestyle='--', alpha=0.4, linewidth=0.8)

# Legend patches
p1 = mpatches.Patch(color=COLORS['secondary'], alpha=0.88, label='Engagement & mapping steps')
p2 = mpatches.Patch(color=COLORS['danger'],    alpha=0.88, label='Adjustment factors (DW / AT / DO / DS)')
p3 = mpatches.Patch(color=COLORS['warning'],   alpha=0.88, label='Full adjustment compliance')
p4 = mpatches.Patch(color=COLORS['primary'],   alpha=0.88, label='Sensitivity analysis')
ax.legend(handles=[p1,p2,p3,p4], loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{OUT}/fig13_calculation_cascade.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/fig13_calculation_cascade.pdf', bbox_inches='tight')
plt.close()
print("✓ Figure 13 saved")

# ─── Figure 14: Adjustment factor co-occurrence matrix ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# LEFT: Co-occurrence heatmap
adj_cols = ['deadweight', 'attribution', 'drop_off', 'displacement']
adj_labels = ['Deadweight', 'Attribution', 'Drop-off', 'Displacement']
cooc = np.zeros((4, 4))
for i, c1 in enumerate(adj_cols):
    for j, c2 in enumerate(adj_cols):
        if i == j:
            cooc[i,j] = m[c1].mean() * 100
        else:
            cooc[i,j] = ((m[c1]==1) & (m[c2]==1)).mean() * 100

ax = axes[0]
im = ax.imshow(cooc, cmap='Blues', aspect='auto', vmin=0, vmax=15)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(adj_labels, fontsize=10)
ax.set_yticklabels(adj_labels, fontsize=10)
ax.set_title('Co-occurrence of Adjustment Factors\n(% of all 383 reports)', fontsize=11, fontweight='bold')

for i in range(4):
    for j in range(4):
        c = 'white' if cooc[i,j] > 8 else '#333333'
        ax.text(j, i, f'{cooc[i,j]:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color=c)
plt.colorbar(im, ax=ax, shrink=0.8, label='% of reports')

# RIGHT: Conditional co-occurrence (given DW, what % have each other?)
ax2 = axes[1]
dw = m[m['deadweight']==1]
cond = {
    '+ Attribution': dw['attribution'].mean()*100,
    '+ Drop-off':    dw['drop_off'].mean()*100,
    '+ Displacement':dw['displacement'].mean()*100,
    '+ Sensitivity': dw['sensitivity'].mean()*100,
    'All four\nadjustments': dw['all_four'].mean()*100,
}
colors2 = [COLORS['secondary'], COLORS['secondary'], COLORS['secondary'],
           COLORS['primary'], COLORS['warning']]
bars2 = ax2.barh(list(cond.keys()), list(cond.values()),
                 color=colors2, alpha=0.85, edgecolor='white', height=0.6)
for bar, val in zip(bars2, cond.values()):
    ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.set_xlabel('% of reports with deadweight (n=39)', fontsize=10)
ax2.set_title('Conditional Compliance:\nGiven Deadweight, what else is present?',
              fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

fig.suptitle('Figure 14: Adjustment Factor Co-occurrence Analysis',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig14_factor_cooccurrence.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/fig14_factor_cooccurrence.pdf', bbox_inches='tight')
plt.close()
print("✓ Figure 14 saved")

# ─── Figure 15: Factor compliance by report type and assurance ───────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# LEFT: Forecast vs Non-forecast
forecast = m[m['report_type_clean']=='Forecast']
non_fc   = m[m['report_type_clean']!='Forecast']

factors_show = ['deadweight','attribution','drop_off','displacement','sensitivity','all_four']
labels_show  = ['Deadweight','Attribution','Drop-off','Displacement','Sensitivity\nanalysis','All four\nadjustments']

fc_vals  = [forecast[c].mean()*100 for c in factors_show]
nfc_vals = [non_fc[c].mean()*100   for c in factors_show]

ax = axes[0]
x = np.arange(len(factors_show))
w = 0.35
b1 = ax.bar(x - w/2, fc_vals,  w, label=f'Forecast (n={len(forecast)})',
            color=COLORS['primary'], alpha=0.85, edgecolor='white')
b2 = ax.bar(x + w/2, nfc_vals, w, label=f'Evaluative/Unknown (n={len(non_fc)})',
            color=COLORS['accent'],  alpha=0.85, edgecolor='white')
for bar, v in zip(b1, fc_vals):
    if v > 1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['primary'])
for bar, v in zip(b2, nfc_vals):
    if v > 1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color='#666666')
ax.set_xticks(x)
ax.set_xticklabels(labels_show, fontsize=9)
ax.set_ylabel('% of reports', fontsize=10)
ax.set_title('Forecast vs. Evaluative Reports:\nAdjustment Factor Compliance', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# RIGHT: Assured vs not-assured factor compliance
assured = m[m['assurance_clean']==1]
notass  = m[m['assurance_clean']==0]

a_vals  = [assured[c].mean()*100 for c in factors_show]
na_vals = [notass[c].mean()*100  for c in factors_show]

ax2 = axes[1]
b3 = ax2.bar(x - w/2, a_vals,  w, label=f'Assured (n={len(assured)})',
             color=COLORS['assured'], alpha=0.85, edgecolor='white')
b4 = ax2.bar(x + w/2, na_vals, w, label=f'Not assured (n={len(notass)})',
             color=COLORS['danger'],  alpha=0.75, edgecolor='white')
for bar, v in zip(b3, a_vals):
    if v > 1:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                 f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['assured'])
for bar, v in zip(b4, na_vals):
    if v > 1:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                 f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['danger'])
ax2.set_xticks(x)
ax2.set_xticklabels(labels_show, fontsize=9)
ax2.set_ylabel('% of reports', fontsize=10)
ax2.set_title('Assured vs. Non-Assured Reports:\nAdjustment Factor Compliance', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Figure 15: Adjustment Factor Compliance by Report Type and Assurance Status',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig15_factors_by_type_assurance.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/fig15_factors_by_type_assurance.pdf', bbox_inches='tight')
plt.close()
print("✓ Figure 15 saved")

# ─── Figure 16: Quality and ratio by factor count ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# LEFT: Quality score by n_factors (0, 1, 2, 3, 4)
ax = axes[0]
factor_groups = m.groupby('n_factors')['quality_pct']
means  = factor_groups.mean()
sems   = factor_groups.sem()
ns     = factor_groups.count()
xs     = means.index
colors3 = [COLORS['danger'] if x==0 else COLORS['warning'] if x==1
           else COLORS['accent'] if x==2 else COLORS['secondary'] if x==3
           else COLORS['primary'] for x in xs]
bars = ax.bar(xs, means.values, color=colors3, alpha=0.88,
              edgecolor='white', linewidth=0.8, width=0.6)
ax.errorbar(xs, means.values, yerr=1.96*sems.values, fmt='none',
            color='#333333', capsize=5, linewidth=1.5)
for x_val, m_val, n_val in zip(xs, means.values, ns.values):
    ax.text(x_val, m_val + 3.5, f'{m_val:.1f}%\n(n={n_val})',
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xs)
ax.set_xticklabels([f'{int(x)} factor{"s" if x!=1 else ""}' for x in xs], fontsize=10)
ax.set_ylabel('Mean quality score (%)', fontsize=10)
ax.set_title('Mean Quality Score\nby Number of Adjustment Factors Applied', fontsize=11, fontweight='bold')
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# RIGHT: SROI ratio by n_factors (box plot)
ax2 = axes[1]
ratio_data = []
for n in [0, 1, 2, 3, 4]:
    sub = m[(m['n_factors']==n) & m['sroi_ratio_value'].notna()]['sroi_ratio_value'].values
    ratio_data.append(sub)

positions = [0,1,2,3,4]
bplot = ax2.boxplot(ratio_data, positions=positions, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', alpha=0.4, markersize=4))
for patch, c in zip(bplot['boxes'], [COLORS['danger'], COLORS['warning'],
                                      COLORS['accent'], COLORS['secondary'], COLORS['primary']]):
    patch.set_facecolor(c)
    patch.set_alpha(0.8)
ax2.set_xticks(positions)
ax2.set_xticklabels([f'{n} factor{"s" if n!=1 else ""}' for n in positions], fontsize=9)
ax2.set_ylabel('SROI ratio', fontsize=10)
ax2.set_title('SROI Ratio Distribution\nby Number of Adjustment Factors Applied', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add n labels and medians
for i, (d, pos) in enumerate(zip(ratio_data, positions)):
    if len(d) > 0:
        ax2.text(pos, ax2.get_ylim()[1]*0.95, f'n={len(d)}', ha='center',
                 fontsize=8, color='#555555')

fig.suptitle('Figure 16: Quality and SROI Ratios by Adjustment Factor Count',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig16_quality_ratio_by_factors.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/fig16_quality_ratio_by_factors.pdf', bbox_inches='tight')
plt.close()
print("✓ Figure 16 saved")

# ─── Summary stats table ──────────────────────────────────────────────────────
summary = {
    'Factor': ['Stakeholder engagement (P1 any score)', 'Theory of change / outcome map',
               'Financial proxies mentioned', 'Deadweight', 'Attribution',
               'Drop-off', 'Displacement', 'DW + AT (both)',
               'All four adjustments', 'Discount rate', 'Sensitivity analysis (P7 any)'],
    'n': [
        m['p1_involve_stakeholders'].gt(0).sum(),
        m['theory_of_change'].sum(),
        m['proxy'].sum(),
        m['deadweight'].sum(),
        m['attribution'].sum(),
        m['drop_off'].sum(),
        m['displacement'].sum(),
        m['dw_and_at'].sum(),
        m['all_four'].sum(),
        m['discount_rate'].sum(),
        m['p7_verify_result'].gt(0).sum(),
    ],
    'pct_of_383': [
        round(m['p1_involve_stakeholders'].gt(0).mean()*100, 1),
        round(m['theory_of_change'].mean()*100, 1),
        round(m['proxy'].mean()*100, 1),
        round(m['deadweight'].mean()*100, 1),
        round(m['attribution'].mean()*100, 1),
        round(m['drop_off'].mean()*100, 1),
        round(m['displacement'].mean()*100, 1),
        round(m['dw_and_at'].mean()*100, 1),
        round(m['all_four'].mean()*100, 1),
        round(m['discount_rate'].mean()*100, 1),
        round(m['p7_verify_result'].gt(0).mean()*100, 1),
    ]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{BASE}/data/sroi_factors_summary.csv', index=False)
print("\n✓ Summary stats table saved to data/sroi_factors_summary.csv")
print(summary_df.to_string())
print("\nAll figures saved to", OUT)
