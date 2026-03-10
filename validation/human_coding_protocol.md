# Human Coding Protocol: SVI Eight Principles Quality Assessment

**Purpose:** Gold-standard validation of keyword and LLM quality scores.
**Task:** Code 10 SROI reports × 8 principles = 80 coding decisions.
**Template:** `human_coding_template.csv` — fill the `human_score` and `human_justification` columns.
**Estimated time:** 3–5 hours for the full 80 decisions.

---

## What You Will Code

For each of 10 SROI reports and each of SVI's 8 principles, you will assign a **score from 0 to 2** based on your expert reading of the full report PDF (accessible via the `url` column).

The template shows you:
- `keyword_score_v2` — what the automated keyword method scored
- `llm_agent_a_score` and `llm_agent_b_score` — what two LLM coders scored
- `key_evidence_identified_by_llm` — the text excerpt the LLM found most relevant

**Your task:** Read the actual report PDF (open the URL) and assign your own independent score, then write a brief justification. Do NOT look at the LLM or keyword scores before forming your judgement — cover those columns and score independently.

---

## Scoring Scale

| Score | Meaning | Threshold |
|-------|---------|-----------|
| **2** | **Substantive compliance** | The report actively demonstrates the principle: it documents specific engagement, provides evidence, shows the method was applied, and does so in a way that would allow a reader to assess quality |
| **1** | **Basic mention / partial compliance** | The principle is addressed at a surface level — mentioned or partially described, but without sufficient detail to assess quality or verify application |
| **0** | **No evidence** | The principle is absent, or mentions are entirely generic (e.g., the word "stakeholders" in the title only) |

---

## Principles — Scoring Criteria

### P1: Involve Stakeholders
**Score 2:** Specific stakeholder groups named + evidence of genuine engagement (e.g., survey methodology described, focus groups conducted, interviews documented, co-production activities described)
**Score 1:** Stakeholders mentioned or listed but engagement process not described
**Score 0:** No substantive mention, or only generic references ("we worked with stakeholders")

### P2: Understand What Changes
**Score 2:** A theory of change, outcome map, or logic model is explicitly presented — showing causal links from activities to outcomes
**Score 1:** Outcomes are listed or described, but without a structured causal framework
**Score 0:** No outcome documentation; only activities or outputs reported

### P3: Value What Matters
**Score 2:** Financial proxies are documented — specific proxy values named (HACT, WELLBY, QALY, unit costs), with sources and rationale
**Score 1:** Mention of monetisation or valuation, but proxies not specified or justified
**Score 0:** No monetisation; or the word "value" used only in the sense of "importance"

### P4: Only Include What Is Material
**Score 2:** The report explicitly discusses materiality — which outcomes were excluded and why; or a structured materiality assessment process is described
**Score 1:** Scope boundaries mentioned (e.g., "this study focuses on...") without explicit materiality reasoning
**Score 0:** No discussion of scope exclusions or materiality

### P5: Do Not Over-claim
**Score 2:** At least two of the four standard adjustments are documented with figures — deadweight, attribution, drop-off, or displacement
**Score 1:** One adjustment mentioned with a figure, or multiple adjustments mentioned without figures
**Score 0:** No discussion of counterfactual, deadweight, attribution, or drop-off; or purely generic ("we tried to be conservative")

**Note on P5:** This is the most technically specific principle. Be strict: a mention of "deadweight" as a concept without an estimated percentage or documented reasoning should score 1 at most.

### P6: Be Transparent
**Score 2:** Key assumptions are explicitly stated — named, numbered, or listed — with sources or rationale for each
**Score 1:** Assumptions acknowledged in general terms, or data sources listed without discussing assumptions
**Score 0:** No discussion of assumptions; or generic statements ("our data is reliable")

### P7: Verify the Result
**Score 2:** A sensitivity analysis is conducted — varying at least one key assumption and reporting alternative ratio results
**Score 1:** Sensitivity analysis mentioned or planned but not executed; or robustness mentioned without documented tests
**Score 0:** No sensitivity analysis

### P8: Be Responsive
**Score 2:** Specific recommendations or key learnings are presented — concrete, actionable, and linked to the SROI findings
**Score 1:** Generic recommendations or lessons noted without direct link to the analysis
**Score 0:** No recommendations or learning section

---

## How to Complete the Template

1. Open `human_coding_template.csv` in Excel or Numbers
2. For each row, open the report PDF via the URL (column 3)
3. Read the report (or skim systematically — see the relevant section for each principle)
4. **Cover columns E–H** before scoring (do not look at keyword or LLM scores)
5. Enter your score (0, 1, or 2) in `human_score`
6. Enter a brief justification (1–2 sentences) in `human_justification`
7. After completing all 80 rows, you may look at the automated scores and note surprises

**Where to look in each report:**
- P1: Introduction, methods, stakeholder section
- P2: Theory of change section, outcome map
- P3: Financial proxies, SROI calculation section
- P4: Scope section, "what we excluded" or "materiality"
- P5: Impact section, adjustment factors table
- P6: Assumptions table or list
- P7: Sensitivity analysis section
- P8: Conclusions, recommendations, or learning section

---

## After Completing the Human Coding

Run the analysis script:

```bash
cd /Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT/Papers/2026_sroi/validation/
python3 human_coding_analysis.py
```

This will compute:
- Cohen's κ and Krippendorff's α: Human vs LLM Agent A, Human vs Keyword
- False positive rate: Keyword>0 when Human=0 (true false positives)
- False negative rate: Keyword=0 when Human>0 (true false negatives)
- Agreement table by principle
- Updated reliability report

---

## Reporting

The human coding results will be reported in the paper as:

> "To establish a gold-standard validation, the lead author independently coded all 80 decisions [10 reports × 8 principles] after the keyword and LLM scoring was complete. Human–LLM agreement was [X]% (Cohen's κ = [X.XX]; Krippendorff's α = [X.XX]). The human coder confirmed [N] of the [954] keyword false positives identified by the LLM validation (false positive confirmation rate: [X]%). Human coding is fully reported in the replication package."
