# LLM Coding Prompts — Replication Materials

**Paper:** "From Principles to Practice: A Systematic Content Analysis of SROI Reporting in the Social Value International Database"

**Purpose:** This document provides the complete prompts and rubric used by the two LLM coding agents (Agent A and Agent B) to score SROI reports on SVI's eight principles. Researchers can use these prompts to replicate the coding, apply the rubric to new reports, or adapt the framework for other content analysis tasks.

---

## Overview of the Coding Framework

The coding framework uses two independent LLM agents operating on the opening excerpt of each SROI report (first 5,000 characters of extracted PDF text):

| Agent | Role | Model | Rubric emphasis |
|-------|------|-------|-----------------|
| Agent A | Systematic coder | `gpt-4o-mini` | Balanced — scores presence of evidence |
| Agent B | Critical reviewer | `gpt-4o-mini` | Conservative — penalises false positives |
| Reconciler | Tie-breaker | `gpt-4o` | Used only when Agent A ≠ Agent B |

All agents use `temperature=0` for deterministic output. Both agents independently receive the report excerpt and principle description; neither sees the other's output before scoring.

---

## Principle Definitions and False Positive Risks

Each principle is operationalised with a description, false positive risk rating, and scoring examples. These definitions are identical across both agents.

---

### P1 — Involve Stakeholders

**Description:** The report demonstrates that stakeholders were meaningfully involved in the SROI process — not just as data sources, but in identifying outcomes and reviewing findings. Evidence includes: named stakeholder groups, consultation activities (interviews, focus groups, workshops), stakeholder validation of the analysis.

**False positive risk:** HIGH — The word "stakeholders" appears in almost all SROI reports as a generic term. Score 1 only if there is ACTIVE engagement evidence, not just mention of stakeholders as a concept.

**Score 1 example:** Report lists stakeholder groups but gives no detail of how they were engaged.
**Score 2 example:** Report describes focus groups with beneficiaries used to validate outcomes.

---

### P2 — Understand What Changes

**Description:** The report documents the causal pathway from activities to outcomes — a theory of change, logic model, or outcome map. Evidence includes: explicit ToC diagram or description, outcome indicators linked to activities, distinction between outputs and outcomes.

**False positive risk:** MEDIUM — "Outcomes", "impact", "change" are generic SROI terms. Score 1 only if outcomes are specifically identified and linked to activities.

**Score 1 example:** Report lists outcomes without documenting how activities lead to them.
**Score 2 example:** Report presents a theory of change diagram with activities → outputs → outcomes.

---

### P3 — Value What Matters

**Description:** The report documents HOW outcomes were monetised — specific financial proxies or unit values used to convert outcomes to money. Evidence includes: named proxy sources (e.g., HACT, WELLBY, government unit cost data), specific £ values assigned to outcomes, justification for proxy selection.

**False positive risk:** VERY HIGH — "Value" appears in every SROI report title and text. Score > 0 ONLY if there is evidence of monetisation proxies — not just use of the word "value".

**Score 1 example:** Report mentions using government cost data but does not specify which.
**Score 2 example:** Report uses HACT Social Value Bank values (£X per outcome Y) with citations.

---

### P4 — Only Include What Is Material

**Description:** The report explicitly discusses which outcomes were excluded and why — a materiality assessment. Evidence includes: outcomes considered but excluded, discussion of scope boundaries, acknowledgement of what is NOT included.

**False positive risk:** MEDIUM — "Scope" and "relevant" are common document words. Score > 0 only if there is an explicit discussion of WHAT WAS EXCLUDED and WHY.

**Score 1 example:** Report mentions "scope" but does not explain what was excluded.
**Score 2 example:** Report lists 3 outcomes considered but excluded as immaterial, with justification.

---

### P5 — Do Not Over-claim

**Description:** The report applies corrections for deadweight (what would have happened anyway), attribution (what share is due to this programme vs others), displacement (did outcomes displace outcomes elsewhere), and/or drop-off (decline in outcomes over time). Evidence includes: named adjustment factors with stated rates.

**False positive risk:** LOW — These are specific technical terms rarely used metaphorically. However, verify that "attribution" refers to causal attribution (not academic citation) and "drop-off" refers to outcome duration (not programme dropout).

**Score 1 example:** Report mentions deadweight is "taken into account" without stating the rate.
**Score 2 example:** Report states deadweight=30%, attribution=20%, drop-off=10% with justification.

---

### P6 — Be Transparent

**Description:** The report documents key assumptions explicitly — not just that assumptions were made, but WHAT they are and WHY. Evidence includes: assumption tables, explicit statements of proxy justification, data source documentation.

**False positive risk:** HIGH — "Data", "evidence", "source" appear in virtually all documents. Score > 0 only if specific assumptions are stated — not just that "data was collected".

**Score 1 example:** Report mentions "data was collected from multiple sources".
**Score 2 example:** Report includes an appendix table listing each assumption and its basis.

---

### P7 — Verify the Result

**Description:** The report tests the sensitivity of the SROI ratio to its key assumptions — a sensitivity analysis, scenario analysis, or range of estimates. Evidence includes: explicit sensitivity table, best/worst case scenarios, discussion of how changes in key assumptions affect the ratio.

**False positive risk:** LOW — "Sensitivity analysis" is specific. However, "conservative estimate" and "scenario" may appear in non-analytical contexts.

**Score 1 example:** Report says findings are "conservative" without testing different assumptions.
**Score 2 example:** Report shows SROI ratio under three scenarios: 3.2:1, 4.4:1, 6.1:1.

---

### P8 — Be Responsive

**Description:** The report includes learning and recommendations — evidence that the SROI process generated organisational learning. Evidence includes: explicit recommendations section, lessons learned, stated changes to programme design, plans for future SROI studies.

**False positive risk:** MEDIUM — "Recommendations" is a common section header. Score 2 only if recommendations are specific and actionable, not boilerplate.

**Score 1 example:** Report includes one generic recommendation ("continue the programme").
**Score 2 example:** Report includes 5 specific recommendations with named responsible parties.

---

## Agent A Prompt Template

**Role:** Systematic content analysis coder. Balanced — scores presence of evidence.

```
You are a systematic content analysis coder specialising in Social Return on Investment (SROI) methodology.

Your task: read the following excerpt from an SROI report and score it on ONE SVI principle.

REPORT TITLE: {title}

PRINCIPLE: {principle_name}
DESCRIPTION: {principle_description}

SCORING RUBRIC:
- Score 0: No evidence that this principle was addressed
- Score 1: Some evidence (mentioned or partially addressed)
- Score 2: Clear, substantive evidence (explicitly documented with specifics)

KEY PHRASE FROM REPORT:
---
{text_excerpt}  ← first 5,000 characters of extracted PDF text
---

Respond ONLY in this exact JSON format (no other text):
{
  "score": <0, 1, or 2>,
  "confidence": <"high", "medium", or "low">,
  "key_phrase": "<exact quote from text that most supports your score, max 80 chars>",
  "justification": "<one sentence explaining your score>"
}
```

---

## Agent B Prompt Template

**Role:** Critical reviewer. Conservative — specifically trained to detect false positives.

```
You are a critical reviewer specialising in Social Return on Investment (SROI) methodology.

Your task: read the following excerpt from an SROI report and score it on ONE SVI principle.
You are specifically trained to detect FALSE POSITIVES — cases where superficial language
looks like compliance but the report does not actually apply the principle rigorously.

REPORT TITLE: {title}

PRINCIPLE: {principle_name}
DESCRIPTION: {principle_description}

FALSE POSITIVE RISK: {false_positive_risk}

SCORING RUBRIC:
- Score 0: No genuine evidence (including: mentions principle in passing, uses word without substance)
- Score 1: Partial evidence (acknowledges the principle but without specific application)
- Score 2: Substantive evidence (specific, documented, verifiable application)

EXAMPLE OF SCORE 1: {score_1_example}
EXAMPLE OF SCORE 2: {score_2_example}

TEXT TO CODE:
---
{text_excerpt}  ← first 5,000 characters of extracted PDF text
---

Respond ONLY in this exact JSON format (no other text):
{
  "score": <0, 1, or 2>,
  "confidence": <"high", "medium", or "low">,
  "false_positive_detected": <true or false>,
  "key_phrase": "<exact quote from text, max 80 chars, or 'none' if score=0>",
  "justification": "<one sentence explaining your score, noting if false positive>"
}
```

---

## Reconciler Prompt Template

**Role:** Senior reconciler. Used only when Agent A and Agent B scores differ. Uses `gpt-4o`.

```
You are the senior reconciler in a content analysis validation team for SROI research.

Two independent coders have disagreed on the score for this report and principle.
Your job: read the text carefully, consider both coders' reasoning, and deliver a final score with full justification.

PRINCIPLE: {principle_name}
DESCRIPTION: {principle_description}
FALSE POSITIVE RISK: {false_positive_risk}

AUTOMATED KEYWORD SCORE: {keyword_score}/2

CODER A (systematic):
  Score: {score_a}/2
  Key phrase: "{key_phrase_a}"
  Justification: {justification_a}

CODER B (critical, false-positive focused):
  Score: {score_b}/2
  Key phrase: "{key_phrase_b}"
  Justification: {justification_b}

TEXT:
---
{text_excerpt}
---

Respond ONLY in this exact JSON format:
{
  "final_score": <0, 1, or 2>,
  "sided_with": <"A", "B", or "neither">,
  "false_positive_confirmed": <true or false>,
  "key_phrase": "<quote or 'none'>",
  "justification": "<2-3 sentence explanation of final decision>"
}
```

---

## Human Coding Protocol

The `human_coding_template.csv` file is designed to support future human expert validation of the automated coding. It contains the same 80 decisions (10 reports × 8 principles) used for the LLM inter-rater reliability sample, with columns for:

- `keyword_score_v2` — the keyword-based upper bound score
- `llm_agent_a_score` — Agent A's score
- `llm_agent_b_score` — Agent B's score
- `key_evidence_identified_by_llm` — the key phrase identified by the LLM
- `human_score` — for human expert entry (currently populated with Agent C scores as a proxy)
- `human_justification` — for human expert rationale

**Applying this to additional reports:** The same two-agent framework can be applied to any SROI report corpus by running `validate_coding.py` with a new JSONL input file. The LLM agents, prompts, and rubric are fully documented here for replication.

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Agent A & B model | `gpt-4o-mini` (`gpt-4o-mini-2024-07-18`) |
| Reconciler model | `gpt-4o` (`gpt-4o-2024-08-06`) |
| Temperature | 0 (deterministic) |
| Max tokens | 300 (Agent A, B); 400 (Reconciler) |
| Text input | First 5,000 characters of extracted PDF text |
| API provider | OpenAI |
| Scoring scale | 0–2 ordinal (0=absent, 1=partial, 2=substantive) |

All prompts, model versions, and temperature settings are fixed and fully reproducible using the cached outputs in `validate_coding.py`.
