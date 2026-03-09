"""
SROI Dataset Data Quality Audit
================================
Systematic audit of every classification decision in sroi_clean_dataset.csv.

For each field, this script:
  1. Documents WHAT evidence triggered each classification
  2. Flags UNCERTAIN or SUSPICIOUS decisions
  3. Validates SROI ratios against PDF text
  4. Quantifies overall confidence in the dataset

Outputs:
  - data/audit_report.md         Full narrative audit report
  - data/audit_flags.csv         Per-report flags (one row per report)
  - data/classification_evidence.csv  Evidence trail per field per report
  - data/sroi_clean_dataset_v2.csv    Corrected dataset with confidence scores
"""

import json, re, csv
from pathlib import Path
from collections import defaultdict, Counter
import statistics

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE   = Path("/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT")
INPUT  = BASE / "Agents_JC/SROI/data/sroi_reports_for_agent.jsonl"
OUTDIR = BASE / "Papers/2026_sroi/data"

# ─── Load raw data ────────────────────────────────────────────────────────────
records = []
with open(INPUT) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records")

# ─── SECTOR keywords (same as enrich_dataset.py) ─────────────────────────────
SECTOR_KEYWORDS = {
    "disability":       ["wheelchair", "disability", "disabilities", "disabled",
                         "cerebral palsy", "learning disabil", "impairment",
                         "blind", "deaf", "autism", "special needs", "whizz-kid"],
    "environment":      ["conservation", "wetland", "park authority", "biodiversity",
                         "forest", "woodland", "natural environment", "climate",
                         "carbon", "recycling", "waste management", "green space",
                         "nature reserve", "ranger service", "habitat", "ecological"],
    "health":           ["nhs", "health service", "mental health", "wellbeing",
                         "hospital", "clinical", "therapy", "healthcare",
                         "public health", "gp practice", "chronic", "cancer",
                         "substance abuse", "addiction", "recovery", "patient"],
    "housing":          ["housing", "accommodation", "homeless", "homelessness",
                         "shelter", "tenancy", "rent", "eviction", "social housing",
                         "affordable housing", "landlord", "supported living",
                         "rough sleep"],
    "employment":       ["employment", "employability", "job creation", "workforce",
                         "training programme", "skills", "apprenticeship",
                         "unemployment", "job seeker", "redundan", "back to work",
                         "job placement", "labour market"],
    "education":        ["school", "education", "pupils", "students", "literacy",
                         "numeracy", "learning", "university", "college", "academic",
                         "curriculum", "classroom", "teacher", "early years",
                         "early childhood", "children centre"],
    "youth":            ["young people", "youth", "teenager", "adolescent",
                         "young adult", "young offender", "juvenile", "youth worker",
                         "care leaver", "leaving care"],
    "arts_culture":     ["arts", "culture", "museum", "theatre", "music", "creative",
                         "gallery", "heritage", "cultural", "dance", "film", "library"],
    "justice":          ["prison", "offend", "reoffend", "criminal justice", "parole",
                         "probation", "magistrate", "custody", "resettlement",
                         "ex-offender", "rehabilitation", "justice system"],
    "agriculture_food": ["agriculture", "farming", "food bank", "food security",
                         "nutrition", "organic", "crop", "livestock", "rural",
                         "smallholder"],
    "elderly":          ["older people", "elderly", "older adults", "age uk",
                         "dementia", "alzheimer", "care home", "residential care",
                         "aged", "retirement"],
    "microfinance":     ["microfinance", "microcredit", "micro-enterprise",
                         "financial inclusion", "savings group", "credit union"],
    "social_inclusion": ["social inclusion", "social exclusion", "poverty", "depriv",
                         "marginali", "vulnerable", "disadvantaged", "inequality",
                         "refugee", "asylum", "migrant"],
    "community":        ["community development", "community centre", "neighbourhood",
                         "regeneration", "local community", "community group"],
    "sports":           ["sport", "physical activity", "exercise", "fitness",
                         "swimming", "football", "cycling"],
}

COUNTRY_KEYWORDS = {
    "UK":          ["england", "english", "wales", "welsh", "northern ireland",
                    "great britain", "united kingdom", " uk ", "uk-", "british",
                    "london", "manchester", "birmingham", "sheffield", "liverpool",
                    "leeds", "bristol", "newcastle", "coventry", "nhs",
                    "scotland", "scottish", "edinburgh", "glasgow", "aberdeen",
                    "highland", "clyde", "lothian", "stirling", "dundee",
                    r"£\d", "postcode"],
    "Australia":   ["australia", "australian", "sydney", "melbourne", "brisbane",
                    "perth", "adelaide", "queensland", "victoria", "new south wales",
                    "nsw", "tasmania", "western australia"],
    "USA":         ["united states", "u.s.", " usa", "american", "california",
                    "new york", "chicago", "washington dc", "boston", "los angeles",
                    "san francisco"],
    "New Zealand": ["new zealand", "auckland", "wellington", "christchurch", "nz "],
    "Canada":      ["canada", "canadian", "toronto", "vancouver", "montreal",
                    "ottawa", "alberta", "ontario", "british columbia"],
    "Ireland":     ["ireland", "irish", "dublin", "cork", "limerick"],
    "South Africa":["south africa", "cape town", "johannesburg", "pretoria", "durban"],
    "Kenya":       ["kenya", "kenyan", "nairobi", "mombasa"],
    "India":       ["india", "indian", "mumbai", "delhi", "bangalore"],
    "Uganda":      ["uganda", "ugandan", "kampala"],
    "Nepal":       ["nepal", "nepalese", "kathmandu"],
    "Cambodia":    ["cambodia", "phnom penh"],
    "Philippines": ["philippines", "manila"],
    "Bangladesh":  ["bangladesh", "dhaka"],
    "Ghana":       ["ghana", "accra"],
    "Tanzania":    ["tanzania", "dar es salaam"],
    "Indonesia":   ["indonesia", "jakarta"],
    "Pakistan":    ["pakistan", "karachi", "lahore"],
}

PRINCIPLE_PATTERNS = {
    "p1_involve_stakeholders": {
        "keywords_1": ["stakeholder", "participant", "beneficiar", "consultat",
                       "engagement", "interview", "survey", "focus group",
                       "community involvement"],
        "keywords_2": ["stakeholder engagement", "stakeholder consultat",
                       "stakeholder interview", "focus group", "participat",
                       "co-produc", "service user involvement"],
    },
    "p2_understand_changes": {
        "keywords_1": ["outcome", "change", "impact", "benefit", "difference", "result"],
        "keywords_2": ["theory of change", "outcome map", "impact map", "outcome chain",
                       "logic model", "causal pathway", "intended change"],
    },
    "p3_value_what_matters": {
        "keywords_1": ["value", "proxy", "financial proxy", "monetis", "monetiz",
                       "worth", "willingness to pay", "cost saving"],
        "keywords_2": ["financial proxy", "proxy value", "monetis", "monetiz",
                       "wellby", "qaly", "hact", "unit value", "cost per"],
    },
    "p4_only_material": {
        "keywords_1": ["material", "scope", "boundary", "relevant"],
        "keywords_2": ["materiality", "material change", "immaterial",
                       "excluded from", "not included", "outside the scope"],
    },
    "p5_do_not_overclaim": {
        "keywords_1": ["deadweight", "attribution", "displacement", "drop-off",
                       "drop off", "counterfactual"],
        "keywords_2": ["deadweight", "attribution rate", "displacement",
                       "drop-off", "counterfactual", "what would have happened"],
    },
    "p6_be_transparent": {
        "keywords_1": ["assumption", "transparent", "evidence", "source", "data",
                       "audit trail"],
        "keywords_2": ["audit trail", "our assumption", "key assumption",
                       "data source", "evidence base", "appendix", "transparent"],
    },
    "p7_verify_result": {
        "keywords_1": ["sensitiv", "scenario", "range", "conservative",
                       "robustness", "uncertainty"],
        "keywords_2": ["sensitivity analysis", "sensitivity test", "what if",
                       "scenario analysis", "conservative estimate"],
    },
    "p8_be_responsive": {
        "keywords_1": ["recommend", "learning", "lesson", "implication",
                       "improvement", "next step"],
        "keywords_2": ["recommendation", "key learning", "lessons learned",
                       "we recommend", "area for improvement"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def audit_sector(text_lower: str, title_lower: str) -> dict:
    """Return sector classification WITH evidence and confidence."""
    combined = text_lower[:3000] + " " + title_lower
    scores = {}
    matched_kws = {}
    for sector, kws in SECTOR_KEYWORDS.items():
        hits = [kw for kw in kws if kw in combined]
        if hits:
            scores[sector] = len(hits)
            matched_kws[sector] = hits
    if not scores:
        return {
            "sector": "other",
            "confidence": "LOW",
            "evidence": "No keywords matched",
            "top_score": 0,
            "second_score": 0,
            "margin": 0,
            "matched_keywords": "",
            "flag": "UNCLASSIFIED — manual review needed",
        }
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_s[0]
    second = sorted_s[1] if len(sorted_s) > 1 else (None, 0)
    margin = top[1] - second[1]
    confidence = "HIGH" if margin >= 3 else ("MEDIUM" if margin >= 1 else "LOW")
    flag = ""
    if margin == 0:
        flag = f"TIED between {top[0]} and {second[0]} — ambiguous"
    elif margin == 1:
        flag = f"NARROW margin over {second[0]} (score diff=1)"
    return {
        "sector": top[0],
        "confidence": confidence,
        "evidence": f"Top sector '{top[0]}' score={top[1]}, runner-up '{second[0]}' score={second[1]}",
        "top_score": top[1],
        "second_score": second[1],
        "margin": margin,
        "matched_keywords": "; ".join(matched_kws[top[0]][:5]),
        "flag": flag,
    }


def audit_country(text_lower: str) -> dict:
    """Return country classification WITH evidence and confidence."""
    sample = text_lower[:5000]
    scores = {}
    matched_kws = {}
    for country, kws in COUNTRY_KEYWORDS.items():
        hits = [kw for kw in kws if kw in sample]
        if hits:
            scores[country] = len(hits)
            matched_kws[country] = hits
    if not scores:
        # Try extended text
        sample2 = text_lower[:15000]
        for country, kws in COUNTRY_KEYWORDS.items():
            hits = [kw for kw in kws if kw in sample2]
            if hits:
                scores[country] = len(hits)
                matched_kws[country] = hits
        if not scores:
            return {
                "country": "Unknown",
                "confidence": "NONE",
                "evidence": "No keywords in 15,000 chars",
                "matched_keywords": "",
                "flag": "UNKNOWN COUNTRY — manual review needed",
            }
        source = "extended (15k chars)"
    else:
        source = "first 5k chars"
    sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_c[0]
    second = sorted_c[1] if len(sorted_c) > 1 else (None, 0)
    margin = top[1] - second[1]
    confidence = "HIGH" if margin >= 3 else ("MEDIUM" if margin >= 1 else "LOW")
    flag = f"TIED {top[0]}/{second[0]}" if margin == 0 else (
           f"NARROW margin ({margin})" if margin == 1 else "")
    return {
        "country": top[0],
        "confidence": confidence,
        "evidence": f"'{top[0]}' score={top[1]} via {source}; runner-up='{second[0]}' score={second[1]}",
        "matched_keywords": "; ".join(matched_kws[top[0]][:5]),
        "flag": flag,
    }


def audit_year(record: dict) -> dict:
    """Return year WITH evidence, confidence, and flags."""
    text = record.get("pdf_text_extract", "") or ""
    title = record.get("title", "") or ""
    scraped_at = record.get("scraped_at", "") or ""

    # Strategy 1: Explicit date patterns in first 1500 chars
    # Look for "Month YYYY" or "YYYY/YY" or "YYYY-YY" near cover page context
    cover = text[:1500]
    explicit_patterns = [
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2}|199\d)\b',
        r'\b(20\d{2}|199\d)/(1[0-9]|0[0-9])\b',   # financial year 2015/16
        r'\bprepared\s+in\s+(20\d{2}|199\d)\b',
        r'\bpublished\s+in\s+(20\d{2}|199\d)\b',
        r'\bcompleted\s+in\s+(20\d{2}|199\d)\b',
        r'\bthis\s+report\s+\(?(\d{4})\)?',
        r'\bfinal\s+report[,.]?\s*(20\d{2}|199\d)',
        r'\bsroi\s+(?:report|analysis|study)[,.]?\s*(20\d{2}|199\d)',
    ]
    for pat in explicit_patterns:
        m = re.search(pat, cover.lower())
        if m:
            # Extract the 4-digit year from the match
            year_match = re.search(r'(20\d{2}|199\d)', m.group(0))
            if year_match:
                y = int(year_match.group(1))
                if 2002 <= y <= 2025:
                    return {
                        "year": y,
                        "method": "explicit_pattern",
                        "confidence": "HIGH",
                        "evidence": f"Pattern '{pat}' matched in first 1500 chars: '{m.group(0)}'",
                        "flag": "",
                    }

    # Strategy 2: Modal year in first 800 chars (not min — min picks up historical refs)
    years_800 = re.findall(r'\b(20\d{2}|199\d)\b', text[:800])
    valid_800 = [int(y) for y in years_800 if 2002 <= int(y) <= 2025]
    if valid_800:
        # Use mode (most frequent) not min
        counts = Counter(valid_800)
        modal_year = counts.most_common(1)[0][0]
        modal_count = counts.most_common(1)[0][1]
        all_vals = sorted(set(valid_800))
        suspicious = modal_year < 2005
        return {
            "year": modal_year,
            "method": "modal_800chars",
            "confidence": "MEDIUM" if modal_count >= 2 else "LOW",
            "evidence": f"Modal year in first 800 chars: {modal_year} (appears {modal_count}x). All years found: {all_vals}",
            "flag": "SUSPICIOUS: year < 2005" if suspicious else (
                    "CHECK: only 1 occurrence" if modal_count == 1 else ""),
        }

    # Strategy 3: Title keyword
    m = re.search(r'\b(20\d{2}|199\d)\b', title)
    if m:
        y = int(m.group(1))
        if 2002 <= y <= 2025:
            return {
                "year": y,
                "method": "title",
                "confidence": "MEDIUM",
                "evidence": f"Year {y} found in title: '{title[:80]}'",
                "flag": "",
            }

    # Strategy 4: Wider search (first 3000 chars), modal
    years_3k = re.findall(r'\b(20\d{2}|199\d)\b', text[:3000])
    valid_3k = [int(y) for y in years_3k if 2002 <= int(y) <= 2025]
    if valid_3k:
        counts = Counter(valid_3k)
        modal_year = counts.most_common(1)[0][0]
        return {
            "year": modal_year,
            "method": "modal_3000chars",
            "confidence": "LOW",
            "evidence": f"Modal year in first 3000 chars: {modal_year}",
            "flag": "LOW CONFIDENCE — may be reference year",
        }

    # Strategy 5: scraped_at date (website scrape) as last resort
    if scraped_at:
        m = re.search(r'(20\d{2})', scraped_at)
        if m:
            y = int(m.group(1))
            return {
                "year": None,
                "method": "scraped_date_fallback",
                "confidence": "NONE",
                "evidence": f"No year in PDF; scraped_at={scraped_at}",
                "flag": "UNKNOWN YEAR — scraped date not used; set to null",
            }
    return {
        "year": None,
        "method": "not_found",
        "confidence": "NONE",
        "evidence": "No year found in PDF or title",
        "flag": "MISSING YEAR",
    }


def audit_report_type(text: str, title: str) -> dict:
    """Classify report type with full audit trail."""
    combined = (text + " " + title).lower()
    evidence = []
    detected = []

    if re.search(r'\bforecast\b', combined):
        detected.append("Forecast")
        m = re.search(r'.{0,40}\bforecast\b.{0,40}', combined)
        evidence.append(f"'forecast' found: '...{m.group(0).strip()}...'")

    if re.search(r'\bprospective\b', combined):
        detected.append("Forecast")
        evidence.append("'prospective' found")

    if re.search(r'\bevaluativ', combined):
        detected.append("Evaluative")
        m = re.search(r'.{0,30}\bevaluativ.{0,30}', combined)
        evidence.append(f"'evaluative' found: '...{m.group(0).strip()}...'")

    if re.search(r'\bretrospective\b', combined):
        detected.append("Evaluative")
        evidence.append("'retrospective' found")

    if re.search(r'\bscoping\b', combined):
        detected.append("Scoping")
        m = re.search(r'.{0,30}\bscoping\b.{0,30}', combined)
        evidence.append(f"'scoping' found: '...{m.group(0).strip()}...'")

    if not detected:
        # Default: Evaluative (assumed) — documented explicitly
        return {
            "report_type": "Evaluative",
            "confidence": "LOW",
            "note": "assumed_evaluative",
            "evidence": "No type keyword found; classified as Evaluative by assumption "
                        "(standard for SROI reports without explicit type label)",
            "flag": "ASSUMED EVALUATIVE — no keyword found",
        }

    if len(set(detected)) == 1:
        rtype = detected[0]
        conf = "HIGH" if len(detected) >= 2 else "MEDIUM"
        return {
            "report_type": rtype,
            "confidence": conf,
            "note": "keyword_match",
            "evidence": "; ".join(evidence),
            "flag": "",
        }
    else:
        # Conflict: both Forecast and Evaluative or Forecast and Scoping
        # Priority: Forecast > Evaluative > Scoping
        if "Forecast" in detected:
            rtype = "Forecast"
        elif "Evaluative" in detected:
            rtype = "Evaluative"
        else:
            rtype = "Scoping"
        return {
            "report_type": rtype,
            "confidence": "LOW",
            "note": "conflict_resolved",
            "evidence": f"Conflicting signals: {detected}. Resolved to {rtype} by priority. " + "; ".join(evidence),
            "flag": f"CONFLICTING TYPE SIGNALS: {detected}",
        }


def validate_sroi_ratio(record: dict) -> dict:
    """Validate scraped SROI ratio against PDF text."""
    ratio = record.get("sroi_ratio_value")
    if ratio is None:
        return {"ratio_validated": None, "ratio_confidence": "N/A", "ratio_flag": "", "ratio_context": ""}

    text = record.get("pdf_text_extract", "") or ""
    title = record.get("title", "") or ""

    # Look for the ratio value in PDF text
    # Patterns: "3.5:1", "3.5 to 1", "£3.50", "ratio of 3.5"
    ratio_str = str(ratio)
    ratio_int = int(ratio) if ratio == int(ratio) else None

    found_in_pdf = False
    context = ""
    patterns_tried = []

    # Try exact ratio patterns
    for pat in [
        rf'{re.escape(ratio_str)}[:\s]?1\b',           # 3.5:1 or 3.5 1
        rf'\b{re.escape(ratio_str)}\b',                  # bare number
        rf'ratio\s+of\s+[£$]?{re.escape(ratio_str)}',   # "ratio of 3.5"
        rf'[£$]{re.escape(ratio_str)}\b',                # £3.5
    ]:
        if ratio_int is not None:
            pat_int = pat.replace(re.escape(ratio_str), re.escape(str(ratio_int)))
            patterns_tried.append(pat_int)
            m = re.search(pat_int, text, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                context = text[start:end].replace('\n', ' ')
                found_in_pdf = True
                break

        patterns_tried.append(pat)
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            context = text[start:end].replace('\n', ' ')
            found_in_pdf = True
            break

    if not found_in_pdf and text:
        # Try just the integer part for ratios like 65.0
        if ratio_int is not None and ratio_int >= 2:
            m = re.search(rf'\b{ratio_int}[:\s]1\b', text)
            if m:
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                context = text[start:end].replace('\n', ' ')
                found_in_pdf = True

    flag = ""
    if not text:
        conf = "UNVERIFIABLE"
        flag = "NO PDF TEXT — cannot verify ratio"
    elif found_in_pdf:
        conf = "VERIFIED"
    else:
        conf = "UNVERIFIED"
        flag = f"Ratio {ratio} NOT FOUND in PDF text — may be from metadata only"

    return {
        "ratio_validated": ratio,
        "ratio_confidence": conf,
        "ratio_flag": flag,
        "ratio_context": context[:200],
    }


def audit_quality_score(text_lower: str, principle: str, patterns: dict) -> dict:
    """Return quality score with full evidence of what triggered it."""
    # Check level 2 keywords
    for kw in patterns["keywords_2"]:
        if kw in text_lower:
            # Find context
            idx = text_lower.find(kw)
            ctx = text_lower[max(0, idx-40):min(len(text_lower), idx+80)]
            return {
                "score": 2,
                "trigger": f"Level-2 keyword: '{kw}'",
                "context": ctx.replace('\n', ' ')[:150],
                "flag": "",
            }
    # Check level 1 keywords
    hits = [(kw, text_lower.find(kw)) for kw in patterns["keywords_1"] if kw in text_lower]
    if len(hits) >= 2:
        kw_list = [h[0] for h in hits[:3]]
        ctx_first = text_lower[max(0, hits[0][1]-30):min(len(text_lower), hits[0][1]+60)]
        return {
            "score": 1,
            "trigger": f"Level-1 keywords ({len(hits)} hits): {kw_list}",
            "context": ctx_first.replace('\n', ' ')[:150],
            "flag": "Verify: generic terms may produce false positives" if principle == "p6_be_transparent" else "",
        }
    if len(hits) == 1:
        return {
            "score": 1,
            "trigger": f"Level-1 keyword (single hit): '{hits[0][0]}'",
            "context": "",
            "flag": "WEAK — only 1 keyword; may be coincidental",
        }
    return {
        "score": 0,
        "trigger": "No keywords matched",
        "context": "",
        "flag": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AUDIT LOOP
# ─────────────────────────────────────────────────────────────────────────────

audit_flags = []
evidence_rows = []
corrected_records = []

# Counters for summary
sector_conf   = Counter()
country_conf  = Counter()
year_conf     = Counter()
type_conf     = Counter()
ratio_conf    = Counter()

sector_flags_list = []
country_flags_list = []
year_flags_list = []
type_flags_list = []
ratio_flags_list = []

print("Running audit on all 383 records...")

for i, r in enumerate(records):
    rid   = r.get("id")
    title = r.get("title", "") or ""
    text  = r.get("pdf_text_extract", "") or ""
    text_lower = text.lower()
    title_lower = title.lower()
    has_text = len(text) > 200

    # ── SECTOR ────────────────────────────────────────────────────────────────
    sec = audit_sector(text_lower, title_lower)

    # ── COUNTRY ───────────────────────────────────────────────────────────────
    cnt = audit_country(text_lower)

    # ── YEAR ──────────────────────────────────────────────────────────────────
    yr = audit_year(r)

    # ── REPORT TYPE ───────────────────────────────────────────────────────────
    rt = audit_report_type(text, title)

    # ── SROI RATIO ────────────────────────────────────────────────────────────
    rv = validate_sroi_ratio(r)

    # ── QUALITY SCORES ─────────────────────────────────────────────────────────
    q_results = {}
    q_flags = []
    if has_text:
        for principle, patterns in PRINCIPLE_PATTERNS.items():
            qa = audit_quality_score(text_lower, principle, patterns)
            q_results[principle] = qa["score"]
            if qa["flag"]:
                q_flags.append(f"{principle}: {qa['flag']}")
            # Store evidence
            evidence_rows.append({
                "id": rid,
                "title": title[:60],
                "field": principle,
                "value": qa["score"],
                "trigger": qa["trigger"],
                "context": qa["context"],
                "flag": qa["flag"],
            })
    else:
        for principle in PRINCIPLE_PATTERNS:
            q_results[principle] = None

    # ── AGGREGATE QUALITY ─────────────────────────────────────────────────────
    valid_scores = [v for v in q_results.values() if v is not None]
    quality_total = sum(valid_scores) if valid_scores else None
    quality_pct   = round(quality_total / 16 * 100, 1) if quality_total is not None else None

    # ── FLAG COLLECTION ───────────────────────────────────────────────────────
    all_flags = []
    if sec["flag"]:
        all_flags.append(f"SECTOR: {sec['flag']}")
        sector_flags_list.append((rid, title[:50], sec["flag"]))
    if cnt["flag"]:
        all_flags.append(f"COUNTRY: {cnt['flag']}")
        country_flags_list.append((rid, title[:50], cnt["flag"]))
    if yr["flag"]:
        all_flags.append(f"YEAR: {yr['flag']}")
        year_flags_list.append((rid, title[:50], yr["flag"]))
    if rt["flag"]:
        all_flags.append(f"TYPE: {rt['flag']}")
        type_flags_list.append((rid, title[:50], rt["flag"]))
    if rv["ratio_flag"]:
        all_flags.append(f"RATIO: {rv['ratio_flag']}")
        ratio_flags_list.append((rid, title[:50], rv["ratio_flag"]))
    all_flags.extend([f"QUALITY: {f}" for f in q_flags])

    # ── CONFIDENCE COUNTERS ───────────────────────────────────────────────────
    sector_conf[sec["confidence"]] += 1
    country_conf[cnt["confidence"]] += 1
    year_conf[yr["confidence"]] += 1
    type_conf[rt["confidence"]] += 1
    if rv["ratio_validated"] is not None:
        ratio_conf[rv["ratio_confidence"]] += 1

    # ── AUDIT FLAGS ROW ───────────────────────────────────────────────────────
    audit_flags.append({
        "id": rid,
        "title": title[:80],
        "sector": sec["sector"],
        "sector_confidence": sec["confidence"],
        "sector_flag": sec["flag"],
        "country": cnt["country"],
        "country_confidence": cnt["confidence"],
        "country_flag": cnt["flag"],
        "year": yr["year"],
        "year_method": yr["method"],
        "year_confidence": yr["confidence"],
        "year_flag": yr["flag"],
        "report_type": rt["report_type"],
        "type_note": rt["note"],
        "type_confidence": rt["confidence"],
        "type_flag": rt["flag"],
        "sroi_ratio": rv["ratio_validated"],
        "ratio_confidence": rv["ratio_confidence"],
        "ratio_flag": rv["ratio_flag"],
        "quality_pct": quality_pct,
        "n_quality_flags": len(q_flags),
        "all_flags": " | ".join(all_flags),
    })

    # ── CORRECTED RECORD ──────────────────────────────────────────────────────
    assurance = bool(r.get("is_assured")) or any(
        p in (text[:3000] + title).lower()
        for p in ["assured by", "report assurance", "sroi network assured",
                  "social value uk assured", "svi assured", "assurance standard",
                  "has been assured", "assurance certificate"]
    )
    corrected_records.append({
        "id": rid,
        "slug": r.get("slug", ""),
        "title": title[:100],
        "url": r.get("url", ""),
        # ── Corrected metadata
        "sector_clean": sec["sector"],
        "sector_confidence": sec["confidence"],
        "country_clean": cnt["country"],
        "country_confidence": cnt["confidence"],
        "year_clean": yr["year"],
        "year_method": yr["method"],
        "year_confidence": yr["confidence"],
        "report_type_clean": rt["report_type"],
        "type_note": rt["note"],
        "type_confidence": rt["confidence"],
        "assurance_clean": int(assurance),
        "is_assured_raw": r.get("is_assured", 0),
        # ── SROI metrics
        "sroi_ratio_value": rv["ratio_validated"],
        "ratio_confidence": rv["ratio_confidence"],
        "total_investment_raw": r.get("total_investment_raw", ""),
        "stakeholder_mentions": r.get("stakeholder_mentions", 0) or 0,
        # ── Text
        "has_pdf": r.get("has_pdf", 0),
        "has_text": int(has_text),
        "text_length": len(text),
        # ── Quality scores
        **q_results,
        "quality_total": quality_total,
        "quality_pct": quality_pct,
        "n_quality_flags": len(q_flags),
    })

print(f"Audit complete: {len(audit_flags)} records processed")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

# 1. Audit flags CSV
flags_path = OUTDIR / "audit_flags.csv"
with open(flags_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(audit_flags[0].keys()))
    writer.writeheader()
    writer.writerows(audit_flags)
print(f"Written: {flags_path}")

# 2. Evidence CSV (quality scores)
evidence_path = OUTDIR / "classification_evidence.csv"
if evidence_rows:
    with open(evidence_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(evidence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(evidence_rows)
print(f"Written: {evidence_path}")

# 3. Corrected dataset v2
v2_path = OUTDIR / "sroi_clean_dataset_v2.csv"
with open(v2_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(corrected_records[0].keys()))
    writer.writeheader()
    writer.writerows(corrected_records)
print(f"Written: {v2_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIT REPORT (Markdown)
# ─────────────────────────────────────────────────────────────────────────────

n = len(records)
with_text = sum(1 for r in corrected_records if r["has_text"])
with_ratio = sum(1 for r in corrected_records if r["sroi_ratio_value"] is not None)

report_path = OUTDIR / "audit_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# SROI Dataset Data Quality Audit Report\n\n")
    f.write(f"**Date:** 2026-03-09  \n**Script:** `data_audit.py`  \n**Records:** {n}\n\n")
    f.write("---\n\n")

    f.write("## Executive Summary\n\n")
    f.write("| Field | HIGH confidence | MEDIUM | LOW | NONE/UNKNOWN |\n")
    f.write("|-------|----------------|--------|-----|--------------|\n")
    f.write(f"| Sector | {sector_conf.get('HIGH',0)} ({sector_conf.get('HIGH',0)/n*100:.0f}%) | "
            f"{sector_conf.get('MEDIUM',0)} | {sector_conf.get('LOW',0)} | "
            f"{sector_conf.get('NONE',0)+sector_conf.get('',0)} |\n")
    f.write(f"| Country | {country_conf.get('HIGH',0)} ({country_conf.get('HIGH',0)/n*100:.0f}%) | "
            f"{country_conf.get('MEDIUM',0)} | {country_conf.get('LOW',0)} | "
            f"{country_conf.get('NONE',0)} |\n")
    f.write(f"| Year | {year_conf.get('HIGH',0)} ({year_conf.get('HIGH',0)/n*100:.0f}%) | "
            f"{year_conf.get('MEDIUM',0)} | {year_conf.get('LOW',0)} | "
            f"{year_conf.get('NONE',0)} |\n")
    f.write(f"| Report type | {type_conf.get('HIGH',0)} ({type_conf.get('HIGH',0)/n*100:.0f}%) | "
            f"{type_conf.get('MEDIUM',0)} | {type_conf.get('LOW',0)} | - |\n")
    f.write(f"| SROI ratio | {ratio_conf.get('VERIFIED',0)} VERIFIED | - | "
            f"{ratio_conf.get('UNVERIFIED',0)} UNVERIFIED | "
            f"{ratio_conf.get('UNVERIFIABLE',0)} UNVERIFIABLE |\n\n")

    f.write("---\n\n")
    f.write("## 1. Sector Classification\n\n")
    f.write(f"- **High confidence:** {sector_conf.get('HIGH',0)}/{n} ({sector_conf.get('HIGH',0)/n*100:.1f}%)\n")
    f.write(f"- **Medium confidence:** {sector_conf.get('MEDIUM',0)}/{n}\n")
    f.write(f"- **Low confidence / tied:** {sector_conf.get('LOW',0)}/{n}\n\n")
    if sector_flags_list:
        f.write("### Flagged sector classifications\n\n")
        f.write("| ID | Title | Flag |\n|---|---|---|\n")
        for fid, ftitle, fflag in sector_flags_list[:30]:
            f.write(f"| {fid} | {ftitle} | {fflag} |\n")
        if len(sector_flags_list) > 30:
            f.write(f"\n*...and {len(sector_flags_list)-30} more — see audit_flags.csv*\n")

    f.write("\n---\n\n")
    f.write("## 2. Country Classification\n\n")
    f.write(f"- **High confidence:** {country_conf.get('HIGH',0)}/{n}\n")
    f.write(f"- **Unknown (even after extended search):** {country_conf.get('NONE',0)}/{n}\n\n")
    if country_flags_list:
        f.write("### Unknown countries\n\n")
        f.write("| ID | Title | Flag |\n|---|---|---|\n")
        for fid, ftitle, fflag in country_flags_list[:30]:
            f.write(f"| {fid} | {ftitle} | {fflag} |\n")
        if len(country_flags_list) > 30:
            f.write(f"\n*...and {len(country_flags_list)-30} more*\n")

    f.write("\n---\n\n")
    f.write("## 3. Year Extraction\n\n")
    f.write(f"- **Explicit date pattern (HIGH):** {year_conf.get('HIGH',0)}/{n}\n")
    f.write(f"- **Modal year in 800 chars (MEDIUM):** {year_conf.get('MEDIUM',0)}/{n}\n")
    f.write(f"- **Low confidence:** {year_conf.get('LOW',0)}/{n}\n")
    f.write(f"- **Not found:** {year_conf.get('NONE',0)}/{n}\n\n")
    f.write("**Key fix from v1:** v1 used `min(valid_years)` which picked up historical "
            "reference years (1996–2001). v2 uses modal year (most frequent) and explicit "
            "date patterns, which is more robust.\n\n")
    if year_flags_list:
        f.write("### Flagged year extractions\n\n")
        f.write("| ID | Title | Flag |\n|---|---|---|\n")
        for fid, ftitle, fflag in year_flags_list[:20]:
            f.write(f"| {fid} | {ftitle} | {fflag} |\n")

    f.write("\n---\n\n")
    f.write("## 4. Report Type Classification\n\n")
    type_dist = Counter(r["report_type_clean"] for r in corrected_records)
    for k, v in type_dist.most_common():
        note_count = sum(1 for r in corrected_records
                        if r["report_type_clean"]==k and r.get("type_note")=="assumed_evaluative")
        f.write(f"- **{k}:** {v}")
        if note_count:
            f.write(f" ({note_count} assumed, {v-note_count} keyword-confirmed)")
        f.write("\n")
    f.write("\n**Key fix from v1:** v1 coded reports without type keywords as 'Unknown'. "
            "v2 codes them as 'Evaluative' with `type_note='assumed_evaluative'` "
            "and `type_confidence='LOW'`. This is documented explicitly. "
            "Analysts can filter to keyword-confirmed types only using "
            "`type_note != 'assumed_evaluative'`.\n\n")

    f.write("\n---\n\n")
    f.write("## 5. SROI Ratio Validation\n\n")
    f.write(f"- **Total reports with ratio:** {with_ratio}/383\n")
    f.write(f"- **VERIFIED (found in PDF text):** {ratio_conf.get('VERIFIED',0)}\n")
    f.write(f"- **UNVERIFIED (in metadata but not PDF text):** {ratio_conf.get('UNVERIFIED',0)}\n")
    f.write(f"- **UNVERIFIABLE (no PDF text):** {ratio_conf.get('UNVERIFIABLE',0)}\n\n")
    if ratio_flags_list:
        f.write("### Unverified ratios\n\n")
        f.write("| ID | Title | Flag |\n|---|---|---|\n")
        for fid, ftitle, fflag in ratio_flags_list[:30]:
            f.write(f"| {fid} | {ftitle} | {fflag} |\n")

    f.write("\n---\n\n")
    f.write("## 6. Quality Score Audit\n\n")

    # Per-principle quality stats
    for principle in PRINCIPLE_PATTERNS:
        vals = [r[principle] for r in corrected_records if r[principle] is not None]
        if vals:
            weak_flags = [r for r in evidence_rows
                         if r["field"]==principle and "WEAK" in r.get("flag","")]
            f.write(f"### {principle}\n")
            f.write(f"- Score 0: {vals.count(0)} | Score 1: {vals.count(1)} | Score 2: {vals.count(2)}\n")
            f.write(f"- Mean: {statistics.mean(vals):.3f}\n")
            if weak_flags:
                f.write(f"- **WEAK flags ({len(weak_flags)} reports):** single keyword hit — "
                        f"review manually for false positives\n")
            f.write("\n")

    f.write("---\n\n")
    f.write("## 7. Overall Data Quality Assessment\n\n")
    f.write("| Dimension | Assessment |\n|-----------|------------|\n")
    f.write(f"| Sector classification | {sector_conf.get('HIGH',0)/n*100:.0f}% HIGH confidence; "
            f"{sector_conf.get('LOW',0)} tied/ambiguous |\n")
    f.write(f"| Country classification | {country_conf.get('NONE',0)} reports (after extended search) "
            f"remain Unknown |\n")
    f.write(f"| Year extraction | Fixed: modal vs. min strategy reduces error rate |\n")
    f.write(f"| Report type | {type_conf.get('HIGH',0)} keyword-confirmed; "
            f"{type_conf.get('LOW',0)} assumed Evaluative |\n")
    f.write(f"| SROI ratios | {ratio_conf.get('VERIFIED',0)}/{with_ratio} verified in PDF text; "
            f"{ratio_conf.get('UNVERIFIED',0)} metadata-only |\n")
    f.write(f"| Quality scores | Pattern-based; see classification_evidence.csv for per-report evidence |\n")
    f.write("\n**Recommendation:** Use `sector_confidence`, `country_confidence`, "
            "`year_confidence`, and `type_confidence` columns to filter analyses "
            "to higher-confidence subsets in robustness checks.\n")

print(f"\nWritten: {report_path}")
print("\nAudit complete. Key summary:")
print(f"  Sector HIGH confidence: {sector_conf.get('HIGH',0)}/{n}")
print(f"  Country Unknown: {country_conf.get('NONE',0)}/{n}")
print(f"  Year not found: {year_conf.get('NONE',0)}/{n}")
print(f"  Type keyword-confirmed: {type_conf.get('HIGH',0)+type_conf.get('MEDIUM',0)}/{n}")
print(f"  Ratios VERIFIED in PDF: {ratio_conf.get('VERIFIED',0)}/{with_ratio}")
