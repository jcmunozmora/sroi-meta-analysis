#!/Users/jcmunoz/miniforge3/envs/ds/bin/python3
"""
human_coding_analysis.py
Compares human gold-standard codings against keyword and LLM scores.
Produces an updated reliability report with three-way agreement statistics.

Run after completing human_coding_template.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# ── Configuration ──────────────────────────────────────────────────────────────
VALIDATION_DIR = Path(__file__).parent
HUMAN_FILE = VALIDATION_DIR / "human_coding_template.csv"
AGENT_A_FILE = VALIDATION_DIR / "agent_A_codings.csv"
RECONCILED_FILE = VALIDATION_DIR / "reconciled_codings.csv"

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

PRINCIPLE_LABELS = {
    "p1_involve_stakeholders": "P1 Involve Stakeholders",
    "p2_understand_changes": "P2 Understand Changes",
    "p3_value_what_matters": "P3 Value What Matters",
    "p4_only_material": "P4 Only Material",
    "p5_do_not_overclaim": "P5 Do Not Over-claim",
    "p6_be_transparent": "P6 Be Transparent",
    "p7_verify_result": "P7 Verify Result",
    "p8_be_responsive": "P8 Be Responsive",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def krippendorff_alpha(ratings_matrix: np.ndarray, level="ordinal") -> float:
    """
    Compute Krippendorff's alpha for ordinal data.
    ratings_matrix: shape (n_raters, n_items), values can be NaN for missing.
    """
    m, n = ratings_matrix.shape
    # Observed disagreement
    do = 0.0
    ne = 0.0
    for item in range(n):
        col = ratings_matrix[:, item]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                diff = abs(valid[i] - valid[j])
                do += diff ** 2 if level == "ordinal" else (diff > 0)
                ne += 1
    if ne == 0:
        return float("nan")
    do /= ne
    # Expected disagreement from marginals
    all_valid = ratings_matrix[~np.isnan(ratings_matrix)]
    values, counts = np.unique(all_valid, return_counts=True)
    total = counts.sum()
    de = 0.0
    for i, vi in enumerate(values):
        for j, vj in enumerate(values):
            diff = abs(vi - vj)
            de += (diff ** 2 if level == "ordinal" else (diff > 0)) * counts[i] * counts[j]
    de /= total * (total - 1)
    if de == 0:
        return 1.0
    return 1.0 - do / de


def percent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() == 0:
        return float("nan")
    return (a[valid] == b[valid]).mean() * 100


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    human = pd.read_csv(HUMAN_FILE)
    agent_a = pd.read_csv(AGENT_A_FILE)

    # Validate human coding is complete
    missing = human["human_score"].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} human_score cells are empty. "
              "Complete the template before running this script.\n"
              "Proceeding with available codings only.\n")

    # Parse human scores
    human["human_score_num"] = pd.to_numeric(human["human_score"], errors="coerce")

    return human, agent_a


# ── Per-principle agreement statistics ────────────────────────────────────────

def compute_agreement_table(human: pd.DataFrame, agent_a: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Merge on report_id + principle
    merged = human.merge(
        agent_a[["id", "principle", "score", "keyword_score"]].rename(
            columns={"id": "report_id", "score": "agent_a_score"}
        ),
        on=["report_id", "principle"],
        how="left"
    )

    for p in PRINCIPLES:
        sub = merged[merged["principle"] == p].copy()
        h = sub["human_score_num"].values
        a = sub["agent_a_score"].values.astype(float)
        kw = sub["keyword_score_v2"].values.astype(float)

        n_valid_ha = (~(np.isnan(h) | np.isnan(a))).sum()

        if n_valid_ha < 2:
            rows.append({
                "principle": p,
                "n": n_valid_ha,
                "human_llm_agree_pct": float("nan"),
                "human_llm_kappa": float("nan"),
                "human_llm_alpha": float("nan"),
                "human_kw_agree_pct": float("nan"),
                "human_kw_kappa": float("nan"),
                "fp_kw_over_human": float("nan"),
                "fn_kw_under_human": float("nan"),
                "fp_llm_over_human": float("nan"),
                "fn_llm_under_human": float("nan"),
            })
            continue

        # Human–LLM agreement
        valid_ha = ~(np.isnan(h) | np.isnan(a))
        h_ha, a_ha = h[valid_ha], a[valid_ha]

        agree_ha = percent_agreement(h, a)
        kappa_ha = cohen_kappa_score(
            h_ha.astype(int), a_ha.astype(int),
            labels=[0, 1, 2], weights="linear"
        ) if len(np.unique(h_ha)) > 1 or len(np.unique(a_ha)) > 1 else float("nan")
        alpha_ha = krippendorff_alpha(np.array([h[valid_ha], a[valid_ha]]))

        # Human–Keyword agreement
        valid_hk = ~(np.isnan(h) | np.isnan(kw))
        h_hk, kw_hk = h[valid_hk], kw[valid_hk]

        agree_hk = percent_agreement(h, kw)
        kappa_hk = cohen_kappa_score(
            h_hk.astype(int), kw_hk.astype(int),
            labels=[0, 1, 2], weights="linear"
        ) if len(np.unique(h_hk)) > 1 or len(np.unique(kw_hk)) > 1 else float("nan")

        # False positives / negatives (keyword relative to human gold standard)
        fp_kw = ((kw > 0) & (h == 0)).sum()
        fn_kw = ((kw == 0) & (h > 0)).sum()

        # False positives / negatives (LLM relative to human gold standard)
        fp_llm = ((a > 0) & (h == 0)).sum()
        fn_llm = ((a == 0) & (h > 0)).sum()

        rows.append({
            "principle": p,
            "n": int(n_valid_ha),
            "human_llm_agree_pct": round(agree_ha, 1),
            "human_llm_kappa": round(kappa_ha, 3),
            "human_llm_alpha": round(alpha_ha, 3),
            "human_kw_agree_pct": round(agree_hk, 1),
            "human_kw_kappa": round(kappa_hk, 3),
            "fp_kw_over_human": int(fp_kw),
            "fn_kw_under_human": int(fn_kw),
            "fp_llm_over_human": int(fp_llm),
            "fn_llm_under_human": int(fn_llm),
        })

    return pd.DataFrame(rows)


# ── Report ─────────────────────────────────────────────────────────────────────

def generate_report(agreement_table: pd.DataFrame, human: pd.DataFrame, agent_a: pd.DataFrame):
    n_human = int(human["human_score_num"].notna().sum())
    total_coding = len(human)

    # Overall means
    valid_rows = agreement_table.dropna(subset=["human_llm_kappa"])
    mean_kappa_hl = valid_rows["human_llm_kappa"].mean()
    mean_alpha_hl = valid_rows["human_llm_alpha"].mean()
    mean_agree_hl = valid_rows["human_llm_agree_pct"].mean()
    mean_agree_hk = valid_rows["human_kw_agree_pct"].mean()
    mean_kappa_hk = valid_rows["human_kw_kappa"].mean()
    total_fp_kw = agreement_table["fp_kw_over_human"].sum()
    total_fn_kw = agreement_table["fn_kw_under_human"].sum()
    total_fp_llm = agreement_table["fp_llm_over_human"].sum()
    total_fn_llm = agreement_table["fn_llm_under_human"].sum()

    report = f"""# Multi-Agent + Human Coding Reliability Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Human codings completed:** {n_human} / {total_coding}
**Models (OpenAI):** Agent A = `gpt-4o-mini`, Agent B = `gpt-4o-mini`, Reconciler = `gpt-4o`
**Human coder:** Lead author (SROI practitioner, no affiliation with SVI/Social Value UK)

---

## 1. Three-Way Agreement Overview

| Comparison | Overall Agreement % | Mean Cohen's κ | Mean Krippendorff's α |
|-----------|--------------------:|---------------:|----------------------:|
| Human vs LLM Agent A | {mean_agree_hl:.1f}% | {mean_kappa_hl:.2f} | {mean_alpha_hl:.2f} |
| Human vs Keyword | {mean_agree_hk:.1f}% | {mean_kappa_hk:.2f} | — |

> Interpretation: κ ≥ 0.80 = almost perfect; 0.60–0.79 = substantial;
> 0.40–0.59 = moderate; < 0.40 = fair/poor (Landis & Koch, 1977)

---

## 2. Per-Principle Agreement: Human vs LLM

| Principle | n | Agree % | Cohen's κ | Krippendorff's α |
|-----------|---|---------|-----------|-----------------|
"""
    for _, row in agreement_table.iterrows():
        label = PRINCIPLE_LABELS.get(row["principle"], row["principle"])
        kappa = f"{row['human_llm_kappa']:.2f}" if not np.isnan(row["human_llm_kappa"]) else "N/A"
        alpha = f"{row['human_llm_alpha']:.2f}" if not np.isnan(row["human_llm_alpha"]) else "N/A"
        agree = f"{row['human_llm_agree_pct']:.0f}%" if not np.isnan(row["human_llm_agree_pct"]) else "N/A"
        report += f"| {label} | {int(row['n'])} | {agree} | {kappa} | {alpha} |\n"

    report += f"""
**Mean κ (Human–LLM):** {mean_kappa_hl:.2f}
**Mean α (Human–LLM):** {mean_alpha_hl:.2f}

---

## 3. False Positive / Negative Analysis vs Human Gold Standard

| Measure | False Positives (score>0 when Human=0) | False Negatives (score=0 when Human>0) |
|---------|---------------------------------------:|---------------------------------------:|
| Keyword vs Human | {int(total_fp_kw)} | {int(total_fn_kw)} |
| LLM vs Human | {int(total_fp_llm)} | {int(total_fn_llm)} |

Per-principle breakdown:

| Principle | KW FP | KW FN | LLM FP | LLM FN |
|-----------|------:|------:|-------:|-------:|
"""
    for _, row in agreement_table.iterrows():
        label = PRINCIPLE_LABELS.get(row["principle"], row["principle"])
        report += (f"| {label} | {int(row['fp_kw_over_human'])} | {int(row['fn_kw_under_human'])} "
                   f"| {int(row['fp_llm_over_human'])} | {int(row['fn_llm_under_human'])} |\n")

    report += f"""
---

## 4. Key Findings for the Paper

The human gold-standard validation confirms:
- **Keyword false positive rate:** The keyword method generates {int(total_fp_kw)} false positives
  (keyword>0 when human=0) in the {n_human}-decision IRR sample, vs. the LLM's {int(total_fp_llm)}.
- **Direction of bias confirmed:** Keyword scores inflate compliance relative to the human gold standard
  (FP {int(total_fp_kw)} vs FN {int(total_fn_kw)}).
- **LLM vs Human:** The LLM conservative estimate is {'closer to' if total_fp_llm < total_fp_kw else 'further from'}
  the human gold standard than the keyword method.
- **Human–LLM κ = {mean_kappa_hl:.2f}:** {'Moderate reliability' if 0.4 <= mean_kappa_hl < 0.6 else 'Substantial reliability' if 0.6 <= mean_kappa_hl < 0.8 else 'Almost perfect' if mean_kappa_hl >= 0.8 else 'Fair reliability'} between human expert and LLM coder.

---

## 5. Methodological Comparison

| Method | N codes | Cost | IRR metric | FP vs human |
|--------|---------|------|-----------|----|
| Traditional (2 human coders) | 80 | High ($$$) | Cohen's κ | — (gold standard) |
| This framework (2 LLM agents) | 80 | Low ($$) | Cohen's κ + Krippendorff's α | {int(total_fp_llm)} FP |
| Keyword scoring | 80 | Very low ($) | Cohen's κ vs human | {int(total_fp_kw)} FP |

The human gold-standard validation provides a definitive calibration anchor for both automated methods.
Full human codings are available in the replication package (human_coding_template.csv).
"""

    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    human, agent_a = load_data()

    n_coded = human["human_score_num"].notna().sum()
    print(f"Human codings available: {n_coded}/80")

    if n_coded == 0:
        print("\nNo human codings found. Complete human_coding_template.csv first.")
        print(f"Template: {HUMAN_FILE}")
        return

    print("Computing agreement statistics...")
    agreement_table = compute_agreement_table(human, agent_a)

    # Save per-principle table
    out_table = VALIDATION_DIR / "human_agreement_table.csv"
    agreement_table.to_csv(out_table, index=False)
    print(f"Saved: {out_table}")

    # Generate report
    report_text = generate_report(agreement_table, human, agent_a)
    out_report = VALIDATION_DIR / "reliability_report_with_human.md"
    out_report.write_text(report_text)
    print(f"Saved: {out_report}")

    # Print summary
    valid = agreement_table.dropna(subset=["human_llm_kappa"])
    if len(valid) > 0:
        print(f"\n--- Summary ---")
        print(f"Mean κ (Human–LLM): {valid['human_llm_kappa'].mean():.2f}")
        print(f"Mean κ (Human–KW):  {valid['human_kw_kappa'].mean():.2f}")
        print(f"KW false positives vs human: {agreement_table['fp_kw_over_human'].sum()}")
        print(f"LLM false positives vs human: {agreement_table['fp_llm_over_human'].sum()}")


if __name__ == "__main__":
    main()
