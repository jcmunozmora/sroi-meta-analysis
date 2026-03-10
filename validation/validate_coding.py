"""
Multi-Agent Content Analysis Validation Framework
===================================================
Validates automated keyword classifications using parallel LLM coding agents.

METHODOLOGICAL CONTRIBUTION
----------------------------
Standard content analysis requires human coders to establish inter-rater
reliability (IRR). At scale (383 reports × 8 principles = 3,064 coding
decisions), this is prohibitive. This framework uses three parallel
LLM agents as independent coders:

  Agent A (Systematic Coder):  First blind coding pass — focused,
                                 principle-by-principle assessment
  Agent B (Critical Coder):    Second pass with devil's advocate framing —
                                 specifically looks for false positives
  Agent C (Reconciler):        Reads A and B, resolves disagreements with
                                 explicit rationale

Agreement between agents establishes LLM-based IRR.
Agreement between agents and automated keyword scores identifies
systematic false positives and negatives in the keyword method.

OUTPUTS
-------
  validation/sample_manifest.csv     Stratified sample selection
  validation/agent_A_codings.csv     Agent A results
  validation/agent_B_codings.csv     Agent B results
  validation/reconciled_codings.csv  Reconciled final codings
  validation/reliability_report.md   Full IRR report with Kappa, etc.
  validation/false_positive_log.csv  Systematic errors in keyword method

USAGE
-----
  export OPENAI_API_KEY=sk-...
  python validation/validate_coding.py --n-sample 60 --principles all
  python validation/validate_coding.py --n-sample 60 --principles p5,p7
  python validation/validate_coding.py --validate-id 9 --principle p5
"""

import json, os, re, csv, time, random, argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env file if present (project-level API key management)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─── Configuration ────────────────────────────────────────────────────────────
BASE    = Path("/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT")
JSONL   = BASE / "Agents_JC/SROI/data/sroi_reports_for_agent.jsonl"
OUTDIR  = BASE / "Papers/2026_sroi/validation"
V2_CSV  = BASE / "Papers/2026_sroi/data/sroi_clean_dataset_v2.csv"

MODEL_FAST     = "gpt-4o-mini"   # Agent A and B (parallel, fast + cheap)
MODEL_CAREFUL  = "gpt-4o"        # Agent C (reconciler, more capable)
MAX_TEXT_CHARS = 4000   # Truncate PDF for LLM (context management)
DELAY_BETWEEN  = 0.5    # seconds between API calls

RANDOM_SEED = 42

# ─── SVI Principle Definitions ────────────────────────────────────────────────
PRINCIPLES = {
    "p1_involve_stakeholders": {
        "name": "Involve Stakeholders",
        "description": (
            "The report demonstrates that stakeholders were meaningfully involved "
            "in the SROI process — not just as data sources, but in identifying "
            "outcomes and reviewing findings. Evidence includes: named stakeholder "
            "groups, consultation activities (interviews, focus groups, workshops), "
            "stakeholder validation of the analysis."
        ),
        "false_positive_risk": (
            "HIGH: The word 'stakeholders' appears in almost all SROI reports as "
            "a generic term. Score 1 only if there is ACTIVE engagement evidence, "
            "not just mention of stakeholders as a concept."
        ),
        "score_1_example": "Report lists stakeholder groups but gives no detail of how they were engaged.",
        "score_2_example": "Report describes focus groups with beneficiaries used to validate outcomes.",
    },
    "p2_understand_changes": {
        "name": "Understand What Changes",
        "description": (
            "The report documents the causal pathway from activities to outcomes — "
            "a theory of change, logic model, or outcome map. Evidence includes: "
            "explicit ToC diagram or description, outcome indicators linked to activities, "
            "distinction between outputs and outcomes."
        ),
        "false_positive_risk": (
            "MEDIUM: 'Outcomes', 'impact', 'change' are generic SROI terms. "
            "Score 1 only if outcomes are specifically identified and linked to activities."
        ),
        "score_1_example": "Report lists outcomes without documenting how activities lead to them.",
        "score_2_example": "Report presents a theory of change diagram with activities → outputs → outcomes.",
    },
    "p3_value_what_matters": {
        "name": "Value What Matters",
        "description": (
            "The report documents HOW outcomes were monetised — specific financial proxies "
            "or unit values used to convert outcomes to money. Evidence includes: "
            "named proxy sources (e.g., HACT, WELLBY, government unit cost data), "
            "specific £ values assigned to outcomes, justification for proxy selection."
        ),
        "false_positive_risk": (
            "VERY HIGH: 'Value' appears in every SROI report title and text. "
            "Score > 0 ONLY if there is evidence of monetisation proxies — "
            "not just use of the word 'value'."
        ),
        "score_1_example": "Report mentions using government cost data but does not specify which.",
        "score_2_example": "Report uses HACT Social Value Bank values (£X per outcome Y) with citations.",
    },
    "p4_only_material": {
        "name": "Only Include What Is Material",
        "description": (
            "The report explicitly discusses which outcomes were excluded and why — "
            "a materiality assessment. Evidence includes: outcomes considered but excluded, "
            "discussion of scope boundaries, acknowledgement of what is NOT included."
        ),
        "false_positive_risk": (
            "MEDIUM: 'Scope' and 'relevant' are common document words. "
            "Score > 0 only if there is an explicit discussion of WHAT WAS EXCLUDED and WHY."
        ),
        "score_1_example": "Report mentions 'scope' but does not explain what was excluded.",
        "score_2_example": "Report lists 3 outcomes considered but excluded as immaterial, with justification.",
    },
    "p5_do_not_overclaim": {
        "name": "Do Not Over-claim",
        "description": (
            "The report applies corrections for deadweight (what would have happened anyway), "
            "attribution (what share is due to this programme vs others), "
            "displacement (did outcomes displace outcomes elsewhere), "
            "and/or drop-off (decline in outcomes over time). "
            "Evidence includes: named adjustment factors with stated rates."
        ),
        "false_positive_risk": (
            "LOW: These are specific technical terms rarely used metaphorically. "
            "However, verify that 'attribution' refers to causal attribution (not academic citation) "
            "and 'drop-off' refers to outcome duration (not programme dropout)."
        ),
        "score_1_example": "Report mentions deadweight is 'taken into account' without stating the rate.",
        "score_2_example": "Report states deadweight=30%, attribution=20%, drop-off=10% with justification.",
    },
    "p6_be_transparent": {
        "name": "Be Transparent",
        "description": (
            "The report documents key assumptions explicitly — not just that assumptions "
            "were made, but WHAT they are and WHY. Evidence includes: assumption tables, "
            "explicit statements of proxy justification, data source documentation."
        ),
        "false_positive_risk": (
            "HIGH: 'Data', 'evidence', 'source' appear in virtually all documents. "
            "Score > 0 only if specific assumptions are stated — not just that 'data was collected'."
        ),
        "score_1_example": "Report mentions 'data was collected from multiple sources'.",
        "score_2_example": "Report includes an appendix table listing each assumption and its basis.",
    },
    "p7_verify_result": {
        "name": "Verify the Result",
        "description": (
            "The report tests the sensitivity of the SROI ratio to its key assumptions — "
            "a sensitivity analysis, scenario analysis, or range of estimates. "
            "Evidence includes: explicit sensitivity table, best/worst case scenarios, "
            "discussion of how changes in key assumptions affect the ratio."
        ),
        "false_positive_risk": (
            "LOW: 'Sensitivity analysis' is specific. However, 'conservative estimate' "
            "and 'scenario' may appear in non-analytical contexts."
        ),
        "score_1_example": "Report says findings are 'conservative' without testing different assumptions.",
        "score_2_example": "Report shows SROI ratio under three scenarios: 3.2:1, 4.4:1, 6.1:1.",
    },
    "p8_be_responsive": {
        "name": "Be Responsive",
        "description": (
            "The report includes learning and recommendations — evidence that the SROI "
            "process generated organisational learning. Evidence includes: explicit "
            "recommendations section, lessons learned, stated changes to programme design, "
            "plans for future SROI studies."
        ),
        "false_positive_risk": (
            "MEDIUM: 'Recommendations' is a common section header. "
            "Score 2 only if recommendations are specific and actionable, not boilerplate."
        ),
        "score_1_example": "Report includes one generic recommendation ('continue the programme').",
        "score_2_example": "Report includes 5 specific recommendations with named responsible parties.",
    },
}

SECTORS = [
    "housing", "education", "employment", "health", "environment",
    "disability", "youth", "arts_culture", "justice", "agriculture_food",
    "elderly", "microfinance", "social_inclusion", "community", "sports", "other"
]


# ─── Prompts ──────────────────────────────────────────────────────────────────

def make_agent_a_prompt(text: str, principle_key: str, title: str) -> str:
    p = PRINCIPLES[principle_key]
    return f"""You are a systematic content analysis coder specialising in Social Return on Investment (SROI) methodology.

Your task: read the following excerpt from an SROI report and score it on ONE SVI principle.

REPORT TITLE: {title}

PRINCIPLE: {p['name']}
DESCRIPTION: {p['description']}

SCORING RUBRIC:
- Score 0: No evidence that this principle was addressed
- Score 1: Some evidence (mentioned or partially addressed)
- Score 2: Clear, substantive evidence (explicitly documented with specifics)

KEY PHRASE FROM REPORT:
---
{text[:MAX_TEXT_CHARS]}
---

Respond ONLY in this exact JSON format (no other text):
{{
  "score": <0, 1, or 2>,
  "confidence": <"high", "medium", or "low">,
  "key_phrase": "<exact quote from text that most supports your score, max 80 chars>",
  "justification": "<one sentence explaining your score>"
}}"""


def make_agent_b_prompt(text: str, principle_key: str, title: str) -> str:
    p = PRINCIPLES[principle_key]
    return f"""You are a critical reviewer specialising in Social Return on Investment (SROI) methodology.

Your task: read the following excerpt from an SROI report and score it on ONE SVI principle.
You are specifically trained to detect FALSE POSITIVES — cases where superficial language
looks like compliance but the report does not actually apply the principle rigorously.

REPORT TITLE: {title}

PRINCIPLE: {p['name']}
DESCRIPTION: {p['description']}

FALSE POSITIVE RISK: {p['false_positive_risk']}

SCORING RUBRIC:
- Score 0: No genuine evidence (including: mentions principle in passing, uses word without substance)
- Score 1: Partial evidence (acknowledges the principle but without specific application)
- Score 2: Substantive evidence (specific, documented, verifiable application)

EXAMPLE OF SCORE 1: {p['score_1_example']}
EXAMPLE OF SCORE 2: {p['score_2_example']}

TEXT TO CODE:
---
{text[:MAX_TEXT_CHARS]}
---

Respond ONLY in this exact JSON format (no other text):
{{
  "score": <0, 1, or 2>,
  "confidence": <"high", "medium", or "low">,
  "false_positive_detected": <true or false>,
  "key_phrase": "<exact quote from text, max 80 chars, or 'none' if score=0>",
  "justification": "<one sentence explaining your score, noting if false positive>"
}}"""


def make_reconciler_prompt(
    text: str, principle_key: str, title: str,
    score_a: int, score_b: int,
    just_a: str, just_b: str, kp_a: str, kp_b: str,
    keyword_score: int
) -> str:
    p = PRINCIPLES[principle_key]
    return f"""You are the senior reconciler in a content analysis validation team for SROI research.

Two independent coders have disagreed on the score for this report and principle.
Your job: read the text carefully, consider both coders' reasoning, and deliver a final score with full justification.

PRINCIPLE: {p['name']}
DESCRIPTION: {p['description']}
FALSE POSITIVE RISK: {p['false_positive_risk']}

AUTOMATED KEYWORD SCORE: {keyword_score}/2

CODER A (systematic):
  Score: {score_a}/2
  Key phrase: "{kp_a}"
  Justification: {just_a}

CODER B (critical, false-positive focused):
  Score: {score_b}/2
  Key phrase: "{kp_b}"
  Justification: {just_b}

TEXT:
---
{text[:MAX_TEXT_CHARS]}
---

Respond ONLY in this exact JSON format:
{{
  "final_score": <0, 1, or 2>,
  "agreement_with": <"A", "B", "neither", "both_partially">,
  "keyword_score_correct": <true or false>,
  "error_type": <"false_positive", "false_negative", "correct", "partial_error">,
  "reconciliation_rationale": "<2-3 sentences explaining final score and what automated method got wrong or right>"
}}"""


def make_sector_prompt(text: str, title: str) -> str:
    sector_list = "\n".join(f"  - {s}" for s in SECTORS)
    return f"""You are classifying an SROI report by sector.

REPORT TITLE: {title}

SECTORS (choose exactly one):
{sector_list}

TEXT (first 1,500 characters):
---
{text[:1500]}
---

Respond ONLY in this exact JSON format:
{{
  "sector": "<one sector from the list above>",
  "confidence": <"high", "medium", or "low">,
  "key_phrase": "<phrase from text that most clearly indicates the sector>",
  "runner_up": "<second most likely sector, or 'none'>",
  "justification": "<one sentence>"
}}"""


# ─── API Call ─────────────────────────────────────────────────────────────────

def call_llm(client, model: str, prompt: str, report_id: int,
             principle: str, agent: str) -> dict:
    """Single OpenAI API call with JSON parsing and error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        # Parse JSON
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        else:
            return {"error": f"No JSON in response: {raw[:100]}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ─── Stratified Sample Selection ─────────────────────────────────────────────

def select_sample(df_v2: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """
    Stratified random sample designed to cover:
    - All confidence levels of quality scores
    - Both Forecast and assumed-Evaluative
    - Range of quality scores (low / medium / high)
    - Reports with and without SROI ratios
    - Mix of sectors and countries
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    strata = {
        "high_quality":   df_v2[df_v2["quality_pct"] >= 60],
        "mid_quality":    df_v2[(df_v2["quality_pct"] >= 30) & (df_v2["quality_pct"] < 60)],
        "low_quality":    df_v2[df_v2["quality_pct"] < 30],
        "forecast":       df_v2[df_v2["report_type_clean"] == "Forecast"],
        "has_ratio":      df_v2[df_v2["sroi_ratio_value"].notna()],
        "assumed_eval":   df_v2[df_v2["type_note"] == "assumed_evaluative"].sample(
                              min(10, len(df_v2[df_v2["type_note"]=="assumed_evaluative"])),
                              random_state=RANDOM_SEED),
    }

    per_stratum = max(3, n // len(strata))
    sampled_ids = set()
    rows = []

    for stratum_name, stratum_df in strata.items():
        available = stratum_df[~stratum_df["id"].isin(sampled_ids)]
        k = min(per_stratum, len(available))
        if k > 0:
            selected = available.sample(k, random_state=RANDOM_SEED)
            for _, row in selected.iterrows():
                sampled_ids.add(row["id"])
                rows.append({**row.to_dict(), "stratum": stratum_name})

    # Fill up to n with random if needed
    remaining = df_v2[~df_v2["id"].isin(sampled_ids)]
    extra_k = max(0, n - len(rows))
    if extra_k > 0:
        extra = remaining.sample(min(extra_k, len(remaining)), random_state=RANDOM_SEED)
        for _, row in extra.iterrows():
            rows.append({**row.to_dict(), "stratum": "random_fill"})

    sample_df = pd.DataFrame(rows).drop_duplicates("id").head(n)
    print(f"Sample selected: {len(sample_df)} reports")
    print(f"  Strata: {sample_df['stratum'].value_counts().to_dict()}")
    print(f"  Quality range: {sample_df['quality_pct'].min():.0f}% – {sample_df['quality_pct'].max():.0f}%")
    print(f"  Forecast: {(sample_df['report_type_clean']=='Forecast').sum()}")
    return sample_df


# ─── Main Validation Pipeline ─────────────────────────────────────────────────

def run_validation(n_sample: int = 60, principles: list = None,
                   single_id: int = None, single_principle: str = None):

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        print("\nRunning in DRY RUN mode — showing what would be validated.")
        dry_run = True
    else:
        client = openai.OpenAI(api_key=api_key)
        dry_run = False

    # Load data
    print("Loading data...")
    records = {}
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r

    df_v2 = pd.read_csv(V2_CSV)
    # Merge with PDF text
    df_v2["pdf_text"] = df_v2["id"].map(
        {rid: r.get("pdf_text_extract", "") or "" for rid, r in records.items()}
    )

    principles_to_validate = principles or list(PRINCIPLES.keys())

    # ── Single report mode ─────────────────────────────────────────────────────
    if single_id is not None:
        row = df_v2[df_v2["id"] == single_id].iloc[0]
        text = row["pdf_text"]
        title = row["title"]
        principle = single_principle or "p5_do_not_overclaim"
        kw_score = row.get(principle, 0)

        print(f"\n=== VALIDATING: {title[:60]} ===")
        print(f"Principle: {principle} | Keyword score: {kw_score}")

        if dry_run:
            print("[DRY RUN] Would call Agent A, Agent B, then Reconciler.")
            return

        prompt_a = make_agent_a_prompt(text, principle, title)
        prompt_b = make_agent_b_prompt(text, principle, title)

        print("Calling Agent A (systematic)...")
        result_a = call_llm(client, MODEL_FAST, prompt_a, single_id, principle, "A")
        time.sleep(DELAY_BETWEEN)

        print("Calling Agent B (critical)...")
        result_b = call_llm(client, MODEL_FAST, prompt_b, single_id, principle, "B")
        time.sleep(DELAY_BETWEEN)

        print(f"\nAgent A: score={result_a.get('score')} | {result_a.get('justification','')}")
        print(f"Agent B: score={result_b.get('score')} | {result_b.get('justification','')} | FP={result_b.get('false_positive_detected')}")
        print(f"Keyword: score={kw_score}")

        if result_a.get("score") != result_b.get("score"):
            print("\nDisagreement → calling Reconciler...")
            prompt_r = make_reconciler_prompt(
                text, principle, title,
                result_a.get("score", 0), result_b.get("score", 0),
                result_a.get("justification", ""), result_b.get("justification", ""),
                result_a.get("key_phrase", ""), result_b.get("key_phrase", ""),
                kw_score
            )
            result_r = call_llm(client, MODEL_CAREFUL, prompt_r, single_id, principle, "R")
            print(f"\nReconciler: final={result_r.get('final_score')} | error_type={result_r.get('error_type')}")
            print(f"  Rationale: {result_r.get('reconciliation_rationale','')}")
        else:
            print(f"\nAgents agree: score={result_a.get('score')}")
        return

    # ── Full sample validation ─────────────────────────────────────────────────
    sample_df = select_sample(df_v2, n_sample)
    sample_path = OUTDIR / "sample_manifest.csv"
    sample_df.drop(columns=["pdf_text"]).to_csv(sample_path, index=False)
    print(f"Sample manifest saved: {sample_path}")

    if dry_run:
        print(f"\n[DRY RUN] Would validate {len(sample_df)} reports × {len(principles_to_validate)} principles")
        print(f"  = {len(sample_df) * len(principles_to_validate)} Agent A calls")
        print(f"  = {len(sample_df) * len(principles_to_validate)} Agent B calls")
        print(f"  ≤ {len(sample_df) * len(principles_to_validate)} Reconciler calls (only on disagreements)")
        n_calls = len(sample_df) * len(principles_to_validate)
        print(f"\nEstimated cost (gpt-4o-mini @ $0.15/1M tokens, ~300 tok/call): ~${n_calls * 2 * 0.15 / 1000:.2f}")
        print(f"Estimated cost (gpt-4o reconciler, 30% disagreement, ~500 tok/call): ~${n_calls * 0.3 * 2.5 / 1000:.2f}")
        print(f"\nTo run: export OPENAI_API_KEY=sk-... && python validation/validate_coding.py")
        return

    results_a, results_b, results_r = [], [], []
    false_positive_log = []
    disagreement_count = 0

    total = len(sample_df) * len(principles_to_validate)
    done = 0

    for _, row in sample_df.iterrows():
        rid = row["id"]
        title = row["title"]
        text = row["pdf_text"]

        for principle in principles_to_validate:
            _kw_val = row.get(principle, 0)
            kw_score = int(_kw_val) if pd.notna(_kw_val) else 0
            done += 1
            print(f"[{done}/{total}] Report {rid} | {principle[:20]} | kw={kw_score}", end="  ")

            # ── Agent A and B in parallel ──────────────────────────────────────
            prompt_a = make_agent_a_prompt(text, principle, title)
            prompt_b = make_agent_b_prompt(text, principle, title)

            with ThreadPoolExecutor(max_workers=2) as ex:
                future_a = ex.submit(call_llm, client, MODEL_FAST, prompt_a, rid, principle, "A")
                future_b = ex.submit(call_llm, client, MODEL_FAST, prompt_b, rid, principle, "B")
                result_a = future_a.result()
                result_b = future_b.result()

            time.sleep(DELAY_BETWEEN)

            score_a = result_a.get("score", 0)
            score_b = result_b.get("score", 0)

            results_a.append({
                "id": rid, "title": title[:60], "principle": principle,
                "score": score_a,
                "confidence": result_a.get("confidence"),
                "key_phrase": result_a.get("key_phrase", ""),
                "justification": result_a.get("justification", ""),
                "keyword_score": kw_score,
            })
            results_b.append({
                "id": rid, "title": title[:60], "principle": principle,
                "score": score_b,
                "confidence": result_b.get("confidence"),
                "false_positive_detected": result_b.get("false_positive_detected", False),
                "key_phrase": result_b.get("key_phrase", ""),
                "justification": result_b.get("justification", ""),
                "keyword_score": kw_score,
            })

            # False positive log
            if result_b.get("false_positive_detected") and kw_score > 0:
                false_positive_log.append({
                    "id": rid, "title": title[:60], "principle": principle,
                    "keyword_score": kw_score,
                    "agent_b_score": score_b,
                    "key_phrase_b": result_b.get("key_phrase", ""),
                    "justification_b": result_b.get("justification", ""),
                })

            # ── Reconciler (only on disagreement) ─────────────────────────────
            if score_a != score_b:
                disagreement_count += 1
                print(f"A={score_a} B={score_b} → RECONCILE", end="")
                prompt_r = make_reconciler_prompt(
                    text, principle, title, score_a, score_b,
                    result_a.get("justification", ""),
                    result_b.get("justification", ""),
                    result_a.get("key_phrase", ""),
                    result_b.get("key_phrase", ""),
                    kw_score
                )
                result_r = call_llm(client, MODEL_CAREFUL, prompt_r, rid, principle, "R")
                time.sleep(DELAY_BETWEEN)
                results_r.append({
                    "id": rid, "title": title[:60], "principle": principle,
                    "score_a": score_a, "score_b": score_b,
                    "final_score": result_r.get("final_score"),
                    "agreement_with": result_r.get("agreement_with"),
                    "keyword_score": kw_score,
                    "keyword_score_correct": result_r.get("keyword_score_correct"),
                    "error_type": result_r.get("error_type"),
                    "reconciliation_rationale": result_r.get("reconciliation_rationale", ""),
                })
                print(f" → final={result_r.get('final_score')} ({result_r.get('error_type')})")
            else:
                print(f"A={score_a} B={score_b} → AGREE")

    # ── Save results ────────────────────────────────────────────────────────
    pd.DataFrame(results_a).to_csv(OUTDIR / "agent_A_codings.csv", index=False)
    pd.DataFrame(results_b).to_csv(OUTDIR / "agent_B_codings.csv", index=False)
    if results_r:
        pd.DataFrame(results_r).to_csv(OUTDIR / "reconciled_codings.csv", index=False)
    if false_positive_log:
        pd.DataFrame(false_positive_log).to_csv(OUTDIR / "false_positive_log.csv", index=False)

    print(f"\nResults saved to {OUTDIR}/")
    print(f"Total disagreements: {disagreement_count}/{total} ({disagreement_count/total*100:.1f}%)")

    # ── Generate reliability report ────────────────────────────────────────
    generate_reliability_report(results_a, results_b, results_r, false_positive_log, total)


# ─────────────────────────────────────────────────────────────────────────────
# INTER-RATER RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────

def cohens_kappa(a: list, b: list) -> float:
    """Compute Cohen's Kappa for two lists of ordinal scores (0/1/2)."""
    n = len(a)
    if n == 0:
        return float("nan")
    # Observed agreement
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement
    categories = [0, 1, 2]
    p_e = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in categories
    )
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha(data: list) -> float:
    """
    Ordinal Krippendorff's alpha for 2 coders.
    data: list of (score_a, score_b) tuples.
    """
    n_pairs = len(data)
    if n_pairs < 2:
        return float("nan")
    # Ordinal difference function: d^2 = (k-l)^2
    D_o = sum((a - b) ** 2 for a, b in data) / n_pairs
    all_vals = [v for pair in data for v in pair]
    mean_val = sum(all_vals) / len(all_vals)
    D_e = sum((v - mean_val) ** 2 for v in all_vals) / (len(all_vals) - 1) * 2
    if D_e == 0:
        return 1.0
    return 1 - D_o / D_e


def generate_reliability_report(results_a, results_b, results_r, fp_log, total):
    """Generate the IRR report in Markdown."""
    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    if df_a.empty:
        print("No results to analyse.")
        return

    report_lines = [
        "# Multi-Agent Coding Reliability Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Total coding decisions:** {total}  ",
        f"**Models (OpenAI):** Agent A = `{MODEL_FAST}`, Agent B = `{MODEL_FAST}`, Reconciler = `{MODEL_CAREFUL}`",
        "",
        "---",
        "",
        "## 1. Overall Agreement",
        "",
    ]

    # Merge A and B
    merged = df_a.merge(df_b, on=["id", "principle"], suffixes=("_a", "_b"))
    n_agree = (merged["score_a"] == merged["score_b"]).sum()
    n_total = len(merged)
    pct_agree = n_agree / n_total * 100 if n_total > 0 else 0

    report_lines += [
        f"- **Overall agent agreement:** {n_agree}/{n_total} ({pct_agree:.1f}%)",
        f"- **Disagreements requiring reconciliation:** {n_total - n_agree} "
        f"({100-pct_agree:.1f}%)",
        "",
        "| Principle | Agreement % | Cohen's κ | Krippendorff's α |",
        "|-----------|------------|-----------|-----------------|",
    ]

    kappa_all, alpha_all = [], []
    for p in PRINCIPLES:
        sub = merged[merged["principle"] == p]
        if len(sub) < 3:
            report_lines.append(f"| {p} | N/A (n<3) | — | — |")
            continue
        scores_a = sub["score_a"].tolist()
        scores_b = sub["score_b"].tolist()
        agree_pct = (sub["score_a"] == sub["score_b"]).mean() * 100
        kappa = cohens_kappa(scores_a, scores_b)
        alpha = krippendorff_alpha(list(zip(scores_a, scores_b)))
        kappa_all.append(kappa)
        alpha_all.append(alpha)
        flag = " ⚠️" if kappa < 0.6 else (" ✓" if kappa >= 0.8 else "")
        report_lines.append(
            f"| {p} | {agree_pct:.0f}% | {kappa:.2f}{flag} | {alpha:.2f} |"
        )

    if kappa_all:
        report_lines += [
            "",
            f"**Mean κ:** {np.mean(kappa_all):.2f}  ",
            f"**Mean α:** {np.mean(alpha_all):.2f}",
            "",
            "> Interpretation: κ ≥ 0.80 = almost perfect; 0.60–0.79 = substantial;",
            "> 0.40–0.59 = moderate; < 0.40 = fair/poor (Landis & Koch, 1977)",
        ]

    # ── Keyword vs. Agent agreement ──────────────────────────────────────────
    report_lines += [
        "",
        "---",
        "",
        "## 2. Keyword Score vs. Agent Validation",
        "",
    ]

    # Use Agent A as primary reference
    kw_vs_a = merged.copy()
    kw_vs_a["kw_correct_A"] = (kw_vs_a["keyword_score_a"] == kw_vs_a["score_a"])
    kw_vs_a["fp_flag"] = (kw_vs_a["keyword_score_a"] > 0) & (kw_vs_a["score_a"] == 0)
    kw_vs_a["fn_flag"] = (kw_vs_a["keyword_score_a"] == 0) & (kw_vs_a["score_a"] > 0)

    overall_agree_kw = kw_vs_a["kw_correct_A"].mean() * 100
    fp_count = kw_vs_a["fp_flag"].sum()
    fn_count = kw_vs_a["fn_flag"].sum()

    report_lines += [
        f"- **Keyword score matches Agent A:** {overall_agree_kw:.1f}%",
        f"- **Keyword false positives** (kw>0, Agent A=0): {fp_count}",
        f"- **Keyword false negatives** (kw=0, Agent A>0): {fn_count}",
        "",
        "| Principle | Keyword–Agent A agree % | False Positives | False Negatives |",
        "|-----------|------------------------|-----------------|-----------------|",
    ]

    for p in PRINCIPLES:
        sub = kw_vs_a[kw_vs_a["principle"] == p]
        if len(sub) < 2:
            continue
        agree_pct = sub["kw_correct_A"].mean() * 100
        fp = sub["fp_flag"].sum()
        fn = sub["fn_flag"].sum()
        flag = " ⚠️" if (fp + fn) > 3 else ""
        report_lines.append(
            f"| {p} | {agree_pct:.0f}%{flag} | {fp} | {fn} |"
        )

    # ── False positive details ────────────────────────────────────────────────
    if fp_log:
        report_lines += [
            "",
            "---",
            "",
            "## 3. Detected False Positives (Agent B)",
            "",
            f"Agent B detected **{len(fp_log)} false positives** — cases where the keyword",
            "method scored >0 but the text does not genuinely apply the principle.",
            "",
            "| ID | Title | Principle | KW Score | Agent B | Key Phrase |",
            "|---|---|---|---|---|---|",
        ]
        for fp in fp_log[:20]:
            report_lines.append(
                f"| {fp['id']} | {fp['title'][:40]} | {fp['principle']} | "
                f"{fp['keyword_score']} | {fp['agent_b_score']} | "
                f"{fp['key_phrase_b'][:40]} |"
            )
        if len(fp_log) > 20:
            report_lines.append(f"\n*...{len(fp_log)-20} more — see false_positive_log.csv*")

    # ── Reconciler outcomes ───────────────────────────────────────────────────
    if results_r:
        df_r = pd.DataFrame(results_r)
        report_lines += [
            "",
            "---",
            "",
            "## 4. Reconciler Outcomes",
            "",
            f"**Total reconciled disagreements:** {len(df_r)}",
            "",
        ]
        if "error_type" in df_r.columns:
            for etype, cnt in df_r["error_type"].value_counts().items():
                report_lines.append(f"- **{etype}:** {cnt} ({cnt/len(df_r)*100:.0f}%)")
        if "keyword_score_correct" in df_r.columns:
            kw_correct = df_r["keyword_score_correct"].sum()
            report_lines += [
                "",
                f"Keyword score confirmed correct by reconciler: "
                f"{kw_correct}/{len(df_r)} disagreed cases ({kw_correct/len(df_r)*100:.0f}%)",
            ]

    # ── Implications for paper ────────────────────────────────────────────────
    report_lines += [
        "",
        "---",
        "",
        "## 5. Implications for Paper",
        "",
        "The multi-agent validation provides four key inputs to the paper:",
        "",
        "1. **IRR statistic (κ and α):** Report these in the Methods section to establish",
        "   coding reliability — standard requirement for content analysis.",
        "2. **False positive rate by principle:** Use to flag which principles have",
        "   higher measurement error, and discuss in Limitations.",
        "3. **Direction of keyword bias:** If FP > FN, keyword scores are inflated",
        "   (reported compliance is an upper bound); if FN > FP, scores are conservative.",
        "4. **Corrected compliance estimates:** Replace keyword scores with",
        "   agent-reconciled scores for the validated sample; compare to full corpus.",
        "",
        "---",
        "",
        "## 6. Methodological Contribution",
        "",
        "This validation framework offers a scalable alternative to traditional",
        "human inter-rater reliability for large-scale content analysis:",
        "",
        "| Method | N codes | Cost | IRR metric |",
        "|--------|---------|------|-----------|",
        f"| Traditional (2 human coders) | {total} | High ($$$) | Cohen's κ |",
        f"| This framework (2 LLM agents) | {total} | Low ($$) | Cohen's κ + Krippendorff's α |",
        f"| Reconciler layer | ~{total//3} (disagreements) | Medium | Error type taxonomy |",
        "",
        "The framework is fully reproducible: all prompts, model versions, and",
        "temperature settings are documented in `validate_coding.py`.",
    ]

    report_path = OUTDIR / "reliability_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReliability report saved: {report_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent SROI coding validation")
    parser.add_argument("--n-sample", type=int, default=60,
                        help="Number of reports to validate (default: 60)")
    parser.add_argument("--principles", type=str, default=None,
                        help="Comma-separated principles, e.g. p5,p7 (default: all)")
    parser.add_argument("--validate-id", type=int, default=None,
                        help="Validate a single report by ID")
    parser.add_argument("--principle", type=str, default=None,
                        help="Single principle for --validate-id mode")
    args = parser.parse_args()

    principles = None
    if args.principles:
        principles = [
            f"p{p.strip()}_" if not p.strip().startswith("p") else p.strip()
            for p in args.principles.split(",")
        ]
        # Normalise to full names
        full_names = {k[:2]: k for k in PRINCIPLES}
        principles = [full_names.get(p[:2], p) for p in principles]

    run_validation(
        n_sample=args.n_sample,
        principles=principles,
        single_id=args.validate_id,
        single_principle=args.principle,
    )
