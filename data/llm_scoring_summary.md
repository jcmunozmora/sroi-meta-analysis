# LLM Quality Scoring — Summary

**Model:** gpt-4o-mini  
**Reports scored:** 383  
**Total coding decisions:** 3064

---

## Keyword vs. LLM Scores by Principle

| Principle | Keyword mean | LLM mean | Difference | Ratio (LLM/KW) |
|-----------|-------------|----------|-----------|---------------|
| Involve Stakeholders | 1.106 | 0.452 | -0.655 | 0.41 ⚠️ |
| Understand What Changes | 1.279 | 0.974 | -0.305 | 0.76 |
| Value What Matters | 0.854 | 0.128 | -0.726 | 0.15 ⚠️ |
| Only Include What Is Material | 0.484 | 0.057 | -0.427 | 0.12 ⚠️ |
| Do Not Over-claim | 0.277 | 0.225 | -0.052 | 0.81 |
| Be Transparent | 1.101 | 0.504 | -0.597 | 0.46 ⚠️ |
| Verify the Result | 0.702 | 0.217 | -0.485 | 0.31 ⚠️ |
| Be Responsive | 0.774 | 0.426 | -0.348 | 0.55 ⚠️ |

**Overall keyword compliance:** 40.4%  
**Overall LLM compliance:** 18.6%  
**Estimated keyword overestimation:** +21.7 percentage points

---

## Confidence Distribution

- **high:** 1997 (65.2%)
- **medium:** 1061 (34.6%)
- **low:** 6 (0.2%)

---

## Keyword vs. LLM Agreement

| Principle | Exact match % | FP (kw>0, llm=0) | FN (kw=0, llm>0) |
|-----------|--------------|-----------------|-----------------|
| Involve Stakeholders | 43% | 127 | 12 |
| Understand What Changes | 69% | 37 | 2 |
| Value What Matters | 36% | 232 | 2 |
| Only Include What Is Material | 61% | 136 | 5 |
| Do Not Over-claim | 72% | 32 | 59 |
| Be Transparent | 38% | 142 | 27 |
| Verify the Result | 53% | 140 | 14 |
| Be Responsive | 46% | 108 | 65 |

---

## Implications for the Paper

The keyword-based quality scores systematically over-estimate compliance.
The LLM-validated scores represent the best available estimate of true compliance.

**Recommended reporting:**
- Keyword-based compliance (v2): **40.4%** (reported in original analysis)
- LLM-validated compliance (v3): **18.6%** (more accurate)

The gap confirms that the principles-practice deficit is even larger than
initially estimated. This strengthens the paper's central argument.