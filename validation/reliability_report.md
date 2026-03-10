# Multi-Agent Coding Reliability Report

**Date:** 2026-03-09  
**Total coding decisions:** 80  
**Models (OpenAI):** Agent A = `gpt-4o-mini`, Agent B = `gpt-4o-mini`, Reconciler = `gpt-4o`

---

## 1. Overall Agreement

- **Overall agent agreement:** 61/80 (76.2%)
- **Disagreements requiring reconciliation:** 19 (23.8%)

| Principle | Agreement % | Cohen's κ | Krippendorff's α |
|-----------|------------|-----------|-----------------|
| p1_involve_stakeholders | 60% | 0.32 ⚠️ | 0.51 |
| p2_understand_changes | 50% | 0.31 ⚠️ | 0.47 |
| p3_value_what_matters | 70% | 0.29 ⚠️ | 0.24 |
| p4_only_material | 90% | 0.74 | 0.75 |
| p5_do_not_overclaim | 70% | 0.29 ⚠️ | 0.24 |
| p6_be_transparent | 90% | 0.78 | 0.79 |
| p7_verify_result | 90% | 0.71 | 0.83 |
| p8_be_responsive | 90% | 0.80 ✓ | 0.81 |

**Mean κ:** 0.53  
**Mean α:** 0.58

> Interpretation: κ ≥ 0.80 = almost perfect; 0.60–0.79 = substantial;
> 0.40–0.59 = moderate; < 0.40 = fair/poor (Landis & Koch, 1977)

---

## 2. Keyword Score vs. Agent Validation

- **Keyword score matches Agent A:** 60.0%
- **Keyword false positives** (kw>0, Agent A=0): 15
- **Keyword false negatives** (kw=0, Agent A>0): 3

| Principle | Keyword–Agent A agree % | False Positives | False Negatives |
|-----------|------------------------|-----------------|-----------------|
| p1_involve_stakeholders | 70% | 0 | 1 |
| p2_understand_changes | 60% | 2 | 0 |
| p3_value_what_matters | 50% ⚠️ | 4 | 0 |
| p4_only_material | 60% | 3 | 0 |
| p5_do_not_overclaim | 70% | 0 | 2 |
| p6_be_transparent | 40% | 2 | 0 |
| p7_verify_result | 70% | 2 | 0 |
| p8_be_responsive | 60% | 2 | 0 |

---

## 3. Detected False Positives (Agent B)

Agent B detected **47 false positives** — cases where the keyword
method scored >0 but the text does not genuinely apply the principle.

| ID | Title | Principle | KW Score | Agent B | Key Phrase |
|---|---|---|---|---|---|
| 370 | Reports Database:Gentoo Living Young Per | p1_involve_stakeholders | 2 | 1 | we will continue to listen, include and  |
| 370 | Reports Database:Gentoo Living Young Per | p2_understand_changes | 2 | 1 | The Theory of Change – How the outcomes  |
| 370 | Reports Database:Gentoo Living Young Per | p3_value_what_matters | 1 | 0 | none |
| 370 | Reports Database:Gentoo Living Young Per | p4_only_material | 1 | 1 | Outcomes excluded pg 33 |
| 370 | Reports Database:Gentoo Living Young Per | p6_be_transparent | 2 | 1 | some statistics about the Young People |
| 370 | Reports Database:Gentoo Living Young Per | p7_verify_result | 2 | 1 | Sensitivity Analysis |
| 370 | Reports Database:Gentoo Living Young Per | p8_be_responsive | 2 | 1 | Reflections and Recommendations |
| 355 | Reports Database:Football Foundation: SR | p2_understand_changes | 2 | 1 | Map outcomes: Identify outcomes and deve |
| 355 | Reports Database:Football Foundation: SR | p3_value_what_matters | 2 | 1 | Evidence outcomes and give them a value. |
| 355 | Reports Database:Football Foundation: SR | p4_only_material | 1 | 1 | establish scope and identify key stakeho |
| 355 | Reports Database:Football Foundation: SR | p6_be_transparent | 1 | 1 | Data collection to find out how many peo |
| 355 | Reports Database:Football Foundation: SR | p8_be_responsive | 2 | 1 | Sharing findings and recommendations. |
| 159 | Reports Database:Social Return on Invest | p1_involve_stakeholders | 2 | 1 | identified following consultation with t |
| 159 | Reports Database:Social Return on Invest | p2_understand_changes | 2 | 1 | Individuals are able to access the healt |
| 159 | Reports Database:Social Return on Invest | p3_value_what_matters | 1 | 0 | none |
| 159 | Reports Database:Social Return on Invest | p4_only_material | 2 | 0 | none |
| 159 | Reports Database:Social Return on Invest | p5_do_not_overclaim | 1 | 0 | none |
| 159 | Reports Database:Social Return on Invest | p6_be_transparent | 2 | 1 | data was collected from multiple sources |
| 159 | Reports Database:Social Return on Invest | p7_verify_result | 2 | 1 | By applying a sensitivity analysis, or v |
| 159 | Reports Database:Social Return on Invest | p8_be_responsive | 2 | 1 | none |

*...27 more — see false_positive_log.csv*

---

## 4. Reconciler Outcomes

**Total reconciled disagreements:** 19

- **false_positive:** 16 (84%)
- **correct:** 3 (16%)

Keyword score confirmed correct by reconciler: 5/19 disagreed cases (26%)

---

## 5. Implications for Paper

The multi-agent validation provides four key inputs to the paper:

1. **IRR statistic (κ and α):** Report these in the Methods section to establish
   coding reliability — standard requirement for content analysis.
2. **False positive rate by principle:** Use to flag which principles have
   higher measurement error, and discuss in Limitations.
3. **Direction of keyword bias:** If FP > FN, keyword scores are inflated
   (reported compliance is an upper bound); if FN > FP, scores are conservative.
4. **Corrected compliance estimates:** Replace keyword scores with
   agent-reconciled scores for the validated sample; compare to full corpus.

---

## 6. Methodological Contribution

This validation framework offers a scalable alternative to traditional
human inter-rater reliability for large-scale content analysis:

| Method | N codes | Cost | IRR metric |
|--------|---------|------|-----------|
| Traditional (2 human coders) | 80 | High ($$$) | Cohen's κ |
| This framework (2 LLM agents) | 80 | Low ($$) | Cohen's κ + Krippendorff's α |
| Reconciler layer | ~26 (disagreements) | Medium | Error type taxonomy |

The framework is fully reproducible: all prompts, model versions, and
temperature settings are documented in `validate_coding.py`.