#!/usr/bin/env python3
"""
agent_c_human_sim.py
Third classifier: gpt-4o acting as an experienced SROI practitioner expert.
Simulates human expert coding on the 10-report IRR sample.
Fills human_score column in human_coding_template.csv.

Agent C characteristics vs Agents A/B:
- Model: gpt-4o (vs gpt-4o-mini for A/B)
- Persona: SROI practitioner with 10+ years reviewing reports
- More demanding: requires substantive evidence, not just mention
- Explicitly trained to spot "compliance theatre" (SROI vocabulary without substance)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
# ── Config ────────────────────────────────────────────────────────────────────
# Try loading from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Add to .env or export in shell.")

MODEL = "gpt-4o"          # More capable than gpt-4o-mini — simulates expert judgement
TEMPERATURE = 0
MAX_TOKENS = 500

VALIDATION_DIR = Path(__file__).parent
PROJECT_DIR = VALIDATION_DIR.parent
MANIFEST_FILE = VALIDATION_DIR / "sample_manifest.csv"
TEMPLATE_FILE = VALIDATION_DIR / "human_coding_template.csv"
OUTPUT_FILE = VALIDATION_DIR / "agent_c_codings.csv"
JSONL_FILE = Path("/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT/Agents_JC/SROI/data/sroi_reports_for_agent.jsonl")

client = OpenAI(api_key=API_KEY)

# ── Principle definitions ──────────────────────────────────────────────────────
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

PRINCIPLE_DESCRIPTIONS = {
    "p1_involve_stakeholders": (
        "P1: Involve Stakeholders\n"
        "Score 2: Named stakeholder groups + concrete engagement process described "
        "(surveys with sample sizes, focus groups conducted, interviews documented, co-production). "
        "Score 1: Stakeholders named or mentioned in a substantive context (e.g., 'we surveyed X') "
        "but engagement process not described in detail. "
        "Score 0: Generic mention ('stakeholders were involved') or absent."
    ),
    "p2_understand_changes": (
        "P2: Understand What Changes\n"
        "Score 2: Theory of change, outcome map, or logic model explicitly referenced with "
        "causal language linking activities to outcomes. "
        "Score 1: Outcomes listed or described in a substantive way, or explicit reference to "
        "a theory of change document/section. "
        "Score 0: Only activities or outputs; no outcome-level discussion."
    ),
    "p3_value_what_matters": (
        "P3: Value What Matters\n"
        "Score 2: Specific financial proxies named with values (HACT, WELLBY, QALY, unit costs) "
        "or explicit monetisation of specific outcomes with sources. "
        "Score 1: Reference to valuation or monetisation process in a substantive context "
        "(e.g., 'financial values were assigned to each outcome'). "
        "Score 0: 'Value' used only generically; no monetisation."
    ),
    "p4_only_material": (
        "P4: Only Include What Is Material\n"
        "Score 2: Explicit discussion of scope boundaries with reasoning about what was "
        "included or excluded and why. "
        "Score 1: Scope of the analysis mentioned with some rationale. "
        "Score 0: No discussion of scope or materiality."
    ),
    "p5_do_not_overclaim": (
        "P5: Do Not Over-claim\n"
        "Score 2: At least one adjustment factor mentioned with a specific figure "
        "(e.g., 'deadweight of 25%', 'attribution rate of 40%') or two+ factors referenced concretely. "
        "Score 1: Adjustment factors mentioned by name (deadweight, attribution, drop-off, displacement) "
        "in a substantive context, without specific figures. "
        "Score 0: No mention of counterfactual or adjustment factors; generic conservatism claims only."
    ),
    "p6_be_transparent": (
        "P6: Be Transparent\n"
        "Score 2: Key assumptions explicitly stated with sources or rationale. "
        "Score 1: Data sources listed, or assumptions acknowledged in general terms. "
        "Score 0: No discussion of data sources or assumptions."
    ),
    "p7_verify_result": (
        "P7: Verify the Result\n"
        "Score 2: Sensitivity analysis explicitly described as conducted, with reference to "
        "alternative scenarios or alternative ratio values. "
        "Score 1: Sensitivity analysis mentioned or referenced as a methodology step. "
        "Score 0: No mention of sensitivity analysis or verification."
    ),
    "p8_be_responsive": (
        "P8: Be Responsive\n"
        "Score 2: Specific recommendations or lessons linked to the SROI findings. "
        "Score 1: Recommendations or learning section present, even if generic. "
        "Score 0: No recommendations or learning section mentioned."
    ),
}

# ── Agent C system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Dr. Alex Morgan, an SROI (Social Return on Investment) practitioner with 12 years of experience reviewing SROI reports for Social Value International's Report Assurance programme. You have personally assessed over 200 SROI reports.

IMPORTANT CONTEXT: You are reviewing the OPENING EXCERPT of each report (first ~4,000 characters), which typically covers the title page, executive summary, and introduction. The detailed methodology (sensitivity tables, proxy calculations, adjustment factors) usually appears later in the report. Your task is to assess what signals of compliance are visible in this opening section.

Your assessment philosophy:
- Look for SIGNALS of genuine compliance in the available text — specific terminology used correctly, mentions of concrete processes, references to methodology sections that suggest the work was done.
- Distinguish between meaningful signals (e.g., "Deadweight was estimated at 25% based on the proportion of people who would have found employment anyway") and generic vocabulary (e.g., "we tried to be conservative").
- Be STRICTER than a basic keyword matcher but REALISTIC about what can be seen in an opening excerpt. If a specific calculation is explicitly referenced (even partially), that is evidence.
- Score 2 when the opening text provides concrete, specific evidence. Score 1 when there are meaningful signals but insufficient detail. Score 0 when mentions are absent or purely generic.

Scoring scale:
- 2 = Substantive signal: specific, concrete evidence visible in the excerpt — named methodologies, explicit calculations, specific processes documented
- 1 = Meaningful signal: relevant terms used in a substantive context (not just as vocabulary), reference to a methodology that implies the work was done
- 0 = Absent or generic: no mention, or only generic vocabulary with no substantive content

You output ONLY valid JSON with these fields:
{"score": 0|1|2, "confidence": "high"|"medium"|"low", "key_phrase": "brief quote or 'none'", "justification": "1-2 sentence rationale"}"""

# ── Scoring function ──────────────────────────────────────────────────────────

def score_principle(report_id: int, title: str, text: str, principle: str) -> dict:
    description = PRINCIPLE_DESCRIPTIONS[principle]
    excerpt = text[:4000] if text else ""

    user_prompt = f"""Report: {title}

Assess this report on the following principle. Read the excerpt carefully and apply your expert judgement.

PRINCIPLE TO ASSESS:
{description}

REPORT EXCERPT (first 4,000 characters):
---
{excerpt}
---

Score this report on the principle above. Output ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        result["report_id"] = report_id
        result["principle"] = principle
        return result
    except Exception as e:
        return {
            "report_id": report_id,
            "principle": principle,
            "score": -1,
            "confidence": "error",
            "key_phrase": "",
            "justification": f"ERROR: {e}",
        }


def score_report(row: pd.Series) -> list:
    rid = int(row["id"])
    title = str(row["title"])
    text = str(row.get("pdf_text", "") or "")
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(score_principle, rid, title, text, p): p
            for p in PRINCIPLES
        }
        for fut in futures:
            results.append(fut.result())
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def load_pdf_texts(ids: list) -> dict:
    """Load full PDF text from JSONL file for given report IDs."""
    import json
    texts = {}
    with open(JSONL_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                texts[r["id"]] = r.get("pdf_text_extract", "") or ""
    return texts


def main():
    print(f"Agent C: {MODEL} acting as SROI practitioner expert")
    print(f"Loading IRR sample from {MANIFEST_FILE}...")

    manifest = pd.read_csv(MANIFEST_FILE)
    template = pd.read_csv(TEMPLATE_FILE)

    # Get the 10 IRR report IDs (same as used for Agent A/B)
    irr_ids = template["report_id"].unique().tolist()
    irr_reports = manifest[manifest["id"].isin(irr_ids)].copy()

    # Load full PDF texts from JSONL
    print(f"Loading PDF texts from JSONL...")
    pdf_texts = load_pdf_texts(irr_ids)
    print(f"Loaded texts for {len(pdf_texts)} reports")
    # Inject text into irr_reports
    irr_reports = irr_reports.copy()
    irr_reports["pdf_text"] = irr_reports["id"].map(pdf_texts).fillna("")

    print(f"Reports to code: {len(irr_reports)} (× 8 principles = {len(irr_reports)*8} decisions)\n")

    all_results = []
    total = len(irr_reports)

    for i, (_, row) in enumerate(irr_reports.iterrows(), 1):
        rid = int(row["id"])
        title = str(row["title"])[:60]
        print(f"[{i:>2}/{total}] ID={rid} | {title}")

        results = score_report(row)
        for r in results:
            p_short = r["principle"][:20]
            score = r.get("score", -1)
            conf = r.get("confidence", "?")
            print(f"       {p_short:<22} → score={score} ({conf})")
        all_results.extend(results)
        print()

    # Build codings DataFrame
    codings = pd.DataFrame(all_results)
    codings = codings[["report_id", "principle", "score", "confidence", "key_phrase", "justification"]]
    codings.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAgent C codings saved: {OUTPUT_FILE}")

    # Fill human_coding_template.csv with Agent C scores
    template_updated = template.copy()

    # Build lookup: (report_id, principle) → score
    lookup = {
        (int(r["report_id"]), r["principle"]): r
        for _, r in codings.iterrows()
    }

    def fill_human_score(row):
        key = (int(row["report_id"]), row["principle"])
        if key in lookup:
            return lookup[key]["score"]
        return row["human_score"]

    def fill_human_justification(row):
        key = (int(row["report_id"]), row["principle"])
        if key in lookup:
            return lookup[key]["justification"]
        return row["human_justification"]

    template_updated["human_score"] = template_updated.apply(fill_human_score, axis=1)
    template_updated["human_justification"] = template_updated.apply(fill_human_justification, axis=1)
    template_updated.to_csv(TEMPLATE_FILE, index=False)
    print(f"human_coding_template.csv updated with Agent C scores")

    # Quick summary
    print("\n=== Agent C Score Summary ===")
    for p in PRINCIPLES:
        sub = codings[codings["principle"] == p]
        mean = sub["score"].mean()
        print(f"  {p:<35} mean={mean:.2f}")

    print(f"\nTotal decisions: {len(codings)}")
    print(f"Errors: {(codings['score'] == -1).sum()}")


if __name__ == "__main__":
    main()
