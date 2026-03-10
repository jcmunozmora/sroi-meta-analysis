#!/Users/jcmunoz/miniforge3/envs/ds/bin/python3
"""
agent_c_full_corpus.py
Runs Agent C (gpt-4o, SROI practitioner persona) on ALL 383 reports.
Adds p1_..._agent_c columns to sroi_clean_dataset_v3.csv.

Features:
- Caching: saves progress to .agent_c_cache.json, safe to interrupt/resume
- Parallel: 8 principles per report processed concurrently
- Rate limiting: 1s delay between reports to avoid API throttling
- Cost estimate: ~3,064 calls × gpt-4o ≈ $30-35 USD

Usage:
    python3 agent_c_full_corpus.py            # run all
    python3 agent_c_full_corpus.py --status   # show cache status only
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")

MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 500

VALIDATION_DIR = Path(__file__).parent
PROJECT_DIR = VALIDATION_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

JSONL_FILE = Path("/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT/Agents_JC/SROI/data/sroi_reports_for_agent.jsonl")
V3_DATASET = DATA_DIR / "sroi_clean_dataset_v3.csv"
CACHE_FILE = VALIDATION_DIR / ".agent_c_cache.json"
OUTPUT_CODINGS = VALIDATION_DIR / "agent_c_full_codings.csv"

client = OpenAI(api_key=API_KEY)

# ── Principles ─────────────────────────────────────────────────────────────────
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


# ── Cache ──────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def cache_key(report_id: int, principle: str) -> str:
    return f"{report_id}__{principle}"


# ── Scoring ────────────────────────────────────────────────────────────────────

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


def score_report(row: pd.Series, cache: dict) -> tuple[list, int]:
    """Score all 8 principles for a report, using cache where available."""
    rid = int(row["id"])
    title = str(row["title"])
    text = str(row.get("pdf_text", "") or "")

    results = []
    to_score = []

    for p in PRINCIPLES:
        k = cache_key(rid, p)
        if k in cache:
            results.append(cache[k])
        else:
            to_score.append(p)

    if not to_score:
        return results, 0  # all cached

    new_results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(score_principle, rid, title, text, p): p
            for p in to_score
        }
        for fut in as_completed(futures):
            new_results.append(fut.result())

    results.extend(new_results)
    return results, len(new_results)


# ── Text loader ────────────────────────────────────────────────────────────────

def load_all_texts() -> dict:
    """Load PDF text for all reports from JSONL."""
    texts = {}
    print(f"Loading PDF texts from {JSONL_FILE}...")
    with open(JSONL_FILE) as f:
        for line in f:
            r = json.loads(line)
            texts[r["id"]] = r.get("pdf_text_extract", "") or ""
    print(f"  Loaded {len(texts)} texts")
    return texts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show cache status only")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N reports (for testing)")
    args = parser.parse_args()

    print("Agent C Full Corpus Scorer")
    print(f"Model: {MODEL} | Cache: {CACHE_FILE.name}")

    cache = load_cache()
    print(f"Cache: {len(cache)} decisions already scored")

    df = pd.read_csv(V3_DATASET)
    print(f"Dataset: {len(df)} reports")

    if args.status:
        scored_reports = set()
        for k in cache:
            rid = k.split("__")[0]
            scored_reports.add(rid)
        print(f"\nCache status: {len(scored_reports)}/{len(df)} reports fully or partially scored")
        print(f"Total decisions cached: {len(cache)} / {len(df) * 8}")
        errors = sum(1 for v in cache.values() if v.get("score") == -1)
        print(f"Errors in cache: {errors}")
        return

    # Load PDF texts
    pdf_texts = load_all_texts()

    # Inject texts
    df["pdf_text"] = df["id"].map(pdf_texts).fillna("")
    reports_with_text = (df["pdf_text"].str.len() > 50).sum()
    print(f"Reports with substantive text: {reports_with_text}/{len(df)}\n")

    target_df = df.head(args.limit) if args.limit else df

    all_results = []
    total = len(target_df)
    api_calls = 0
    errors = 0

    start = time.time()

    for i, (_, row) in enumerate(target_df.iterrows(), 1):
        rid = int(row["id"])
        title = str(row["title"])[:60]

        results, new_calls = score_report(row, cache)
        api_calls += new_calls

        for r in results:
            k = cache_key(rid, r["principle"])
            cache[k] = r
            if r.get("score") == -1:
                errors += 1

        all_results.extend(results)

        # Print progress
        cached_marker = " [cached]" if new_calls == 0 else f" [{new_calls} API calls]"
        scores_str = " ".join(
            str(next((r["score"] for r in results if r["principle"] == p), "?"))
            for p in PRINCIPLES
        )
        elapsed = time.time() - start
        eta_s = (elapsed / i) * (total - i) if i > 1 else 0
        eta_min = eta_s / 60
        print(f"[{i:>3}/{total}] ID={rid:>4} | {title:<40} | {scores_str}{cached_marker} | ETA {eta_min:.1f}m")

        # Save cache every 10 reports
        if i % 10 == 0:
            save_cache(cache)

        # Small delay between reports to avoid rate limits
        if new_calls > 0:
            time.sleep(0.5)

    save_cache(cache)
    print(f"\nTotal API calls this run: {api_calls}")
    print(f"Total errors: {errors}")

    # ── Build output codings CSV ───────────────────────────────────────────────
    codings = pd.DataFrame(all_results)
    codings = codings[["report_id", "principle", "score", "confidence", "key_phrase", "justification"]]
    codings.to_csv(OUTPUT_CODINGS, index=False)
    print(f"\nFull codings saved: {OUTPUT_CODINGS}")

    # ── Merge into v3 dataset ──────────────────────────────────────────────────
    print("\nMerging Agent C scores into v3 dataset...")

    # Pivot codings to wide format
    pivot = codings[codings["score"] >= 0].pivot_table(
        index="report_id", columns="principle", values="score", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={
        p: f"{p}_agent_c" for p in PRINCIPLES
    })
    pivot = pivot.rename(columns={"report_id": "id"})

    # Add total and pct columns
    agent_c_cols = [f"{p}_agent_c" for p in PRINCIPLES]
    pivot["quality_total_agent_c"] = pivot[agent_c_cols].sum(axis=1)
    pivot["quality_pct_agent_c"] = pivot["quality_total_agent_c"] / 16 * 100

    # Remove existing agent_c columns from df if present (avoid duplicates)
    existing_ac_cols = [c for c in df.columns if "agent_c" in c]
    if existing_ac_cols:
        df = df.drop(columns=existing_ac_cols)

    # Merge
    df_out = df.drop(columns=["pdf_text"]).merge(pivot, on="id", how="left")
    df_out.to_csv(V3_DATASET, index=False)
    print(f"v3 dataset updated: {V3_DATASET}")
    print(f"New columns added: {agent_c_cols + ['quality_total_agent_c', 'quality_pct_agent_c']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Agent C Score Summary (all reports) ===")
    valid = codings[codings["score"] >= 0]
    for p in PRINCIPLES:
        sub = valid[valid["principle"] == p]
        mean_score = sub["score"].mean()
        n = len(sub)
        print(f"  {p:<35} n={n:>3}  mean={mean_score:.3f}")

    print(f"\nOverall mean score: {valid['score'].mean():.3f}")
    print(f"Overall compliance %: {valid['score'].mean() / 2 * 100:.1f}%")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
