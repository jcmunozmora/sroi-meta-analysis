"""
SROI Reports Dataset Enrichment
================================
Extracts structured metadata and SVI quality scores from PDF text
of the 383 reports in the Social Value UK database.

Input:  sroi_reports_for_agent.jsonl
Output: sroi_clean_dataset.csv, sroi_ratios_subset.csv, enrichment_summary.txt
"""

import json
import csv
import re
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/Users/jcmunoz/Library/CloudStorage/OneDrive-UniversidadEAFIT")
INPUT  = BASE / "Agents_JC/SROI/data/sroi_reports_for_agent.jsonl"
OUTDIR = BASE / "Papers/2026_sroi/data"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─── Sector classification keywords ───────────────────────────────────────────
SECTOR_KEYWORDS = {
    "disability":        ["wheelchair", "disability", "disabilities", "disabled", "mobility equipment",
                          "cerebral palsy", "learning disabil", "impairment", "blind", "deaf", "autism",
                          "special needs", "whizz-kid"],
    "environment":       ["conservation", "wetland", "park authority", "biodiversity", "forest", "woodland",
                          "natural environment", "climate", "carbon", "recycling", "waste management",
                          "green space", "nature reserve", "ranger service", "habitat", "ecological"],
    "health":            ["nhs", "health service", "mental health", "wellbeing", "hospital", "clinical",
                          "therapy", "healthcare", "public health", "gp practice", "chronic", "cancer",
                          "substance abuse", "addiction", "recovery", "patient"],
    "housing":           ["housing", "accommodation", "homeless", "homelessness", "shelter", "tenancy",
                          "rent", "eviction", "social housing", "affordable housing", "landlord",
                          "supported living", "rough sleep"],
    "employment":        ["employment", "employability", "job creation", "workforce", "training programme",
                          "skills", "apprenticeship", "unemployment", "job seeker", "redundan",
                          "back to work", "job placement", "labour market"],
    "education":         ["school", "education", "pupils", "students", "literacy", "numeracy", "learning",
                          "university", "college", "academic", "curriculum", "classroom", "teacher",
                          "early years", "early childhood", "children centre"],
    "youth":             ["young people", "youth", "teenager", "adolescent", "young adult",
                          "young offender", "juvenile", "youth worker", "care leaver", "leaving care"],
    "arts_culture":      ["arts", "culture", "museum", "theatre", "music", "creative", "gallery",
                          "heritage", "cultural", "dance", "sport", "leisure", "film", "library"],
    "justice":           ["prison", "offend", "reoffend", "criminal justice", "parole", "probation",
                          "magistrate", "custody", "resettlement", "ex-offender", "rehabilitation",
                          "justice system"],
    "agriculture_food":  ["agriculture", "farming", "food", "farm", "crop", "livestock", "rural",
                          "smallholder", "food bank", "food security", "nutrition", "organic"],
    "elderly":           ["older people", "elderly", "older adults", "age uk", "dementia", "alzheimer",
                          "care home", "residential care", "aged", "retirement"],
    "microfinance":      ["microfinance", "microcredit", "micro-enterprise", "microenterprise",
                          "small business loan", "financial inclusion", "savings group", "credit union"],
    "social_inclusion":  ["social inclusion", "social exclusion", "poverty", "depriv", "marginali",
                          "vulnerable", "disadvantaged", "inequality", "isolated", "community cohesion",
                          "refugee", "asylum", "migrant"],
    "community":         ["community development", "community centre", "neighbourhood", "regeneration",
                          "local community", "community group", "volunteer"],
    "sports":            ["sport", "physical activity", "exercise", "active", "fitness", "swimming",
                          "football", "cycling", "walking"],
}

# ─── Country classification ────────────────────────────────────────────────────
COUNTRY_KEYWORDS = {
    "UK":           ["england", "english", "wales", "welsh", "northern ireland", "great britain",
                     "united kingdom", " uk ", "uk-", "british", "london", "manchester", "birmingham",
                     "sheffield", "liverpool", "leeds", "bristol", "newcastle", "coventry",
                     "reading, uk", "scotland", "scottish", "edinburgh", "glasgow", "aberdeen",
                     "highland", "clyde", "lothian", "stirling", "dundee"],
    "Australia":    ["australia", "australian", "sydney", "melbourne", "brisbane", "perth",
                     "adelaide", "queensland", "victoria", "new south wales", "nsw", "tasmania",
                     "western australia"],
    "USA":          ["united states", "u.s.", " usa", "american", "california", "new york",
                     "chicago", "washington dc", "boston", "los angeles", "san francisco",
                     "federal", "state of "],
    "New Zealand":  ["new zealand", "auckland", "wellington", "christchurch", "nz ", " nz"],
    "Canada":       ["canada", "canadian", "toronto", "vancouver", "montreal", "ottawa", "alberta",
                     "ontario", "british columbia"],
    "Ireland":      ["ireland", "irish", "dublin", "cork", "limerick", " roi "],
    "South Africa": ["south africa", "south african", "cape town", "johannesburg", "pretoria",
                     "durban", "gauteng"],
    "Kenya":        ["kenya", "kenyan", "nairobi", "mombasa"],
    "India":        ["india", "indian", "mumbai", "delhi", "bangalore", "kolkata"],
    "Uganda":       ["uganda", "ugandan", "kampala"],
    "Cambodia":     ["cambodia", "cambodian", "phnom penh"],
    "Hong Kong":    ["hong kong"],
    "Ecuador":      ["ecuador", "ecuadorian", "quito"],
    "Indonesia":    ["indonesia", "indonesian", "jakarta"],
    "Ghana":        ["ghana", "ghanaian", "accra"],
    "Malawi":       ["malawi", "malawian", "lilongwe"],
    "Tanzania":     ["tanzania", "tanzanian", "dar es salaam"],
    "Philippines":  ["philippines", "philippine", "manila"],
    "Pakistan":     ["pakistan", "pakistani", "karachi", "lahore", "islamabad"],
    "Bangladesh":   ["bangladesh", "bangladeshi", "dhaka"],
    "Nepal":        ["nepal", "nepalese", "kathmandu"],
}

# ─── Report type ──────────────────────────────────────────────────────────────
def classify_report_type(text: str, title: str) -> str:
    combined = (text + " " + title).lower()
    if re.search(r'\bforecast\b', combined):
        return "Forecast"
    if re.search(r'\bevaluativ\b', combined):
        return "Evaluative"
    if re.search(r'\bscoping\b', combined):
        return "Scoping"
    if re.search(r'\bprospective\b', combined):
        return "Forecast"
    if re.search(r'\bretrospective\b', combined):
        return "Evaluative"
    return "Unknown"


# ─── SVI 8-Principles Quality Scoring ────────────────────────────────────────
# Each principle scored 0 (absent), 1 (mentioned), 2 (substantive evidence)

PRINCIPLE_PATTERNS = {
    # P1: Involve stakeholders
    "p1_involve_stakeholders": {
        "keywords_1": ["stakeholder", "participant", "beneficiar", "consultat", "engagement",
                       "interview", "survey", "focus group", "community involvement"],
        "keywords_2": ["stakeholder engagement", "stakeholder consultat", "stakeholder interview",
                       "focus group", "participat", "co-produc", "service user involvement"]
    },
    # P2: Understand what changes
    "p2_understand_changes": {
        "keywords_1": ["outcome", "change", "impact", "benefit", "difference", "result"],
        "keywords_2": ["theory of change", "outcome map", "impact map", "outcome chain",
                       "logic model", "causal pathway", "intended change"]
    },
    # P3: Value what matters (proxies/monetisation)
    "p3_value_what_matters": {
        "keywords_1": ["value", "proxy", "financial proxy", "monetis", "monetiz", "worth",
                       "willingness to pay", "cost saving"],
        "keywords_2": ["financial proxy", "proxy value", "monetis", "monetiz",
                       "wellby", "qaly", "hact", "unit value", "cost per"]
    },
    # P4: Only include what is material
    "p4_only_material": {
        "keywords_1": ["material", "material", "scope", "boundary", "relevant"],
        "keywords_2": ["materiality", "material change", "immaterial", "excluded from",
                       "not included", "outside the scope"]
    },
    # P5: Do not over-claim (deadweight, attribution, displacement, drop-off)
    "p5_do_not_overclaim": {
        "keywords_1": ["deadweight", "attribution", "displacement", "drop-off", "drop off",
                       "counterfactual"],
        "keywords_2": ["deadweight", "attribution rate", "displacement", "drop-off",
                       "counterfactual", "what would have happened"]
    },
    # P6: Be transparent
    "p6_be_transparent": {
        "keywords_1": ["assumption", "transparent", "evidence", "source", "data",
                       "audit trail"],
        "keywords_2": ["audit trail", "our assumption", "key assumption", "data source",
                       "evidence base", "appendix", "transparent"]
    },
    # P7: Verify the result (sensitivity analysis)
    "p7_verify_result": {
        "keywords_1": ["sensitiv", "scenario", "range", "conservative", "robustness",
                       "uncertainty"],
        "keywords_2": ["sensitivity analysis", "sensitivity test", "what if", "scenario analysis",
                       "conservative estimate", "optimistic", "pessimistic"]
    },
    # P8: Be responsive (recommendations, learning)
    "p8_be_responsive": {
        "keywords_1": ["recommend", "learning", "lesson", "implication", "improvement",
                       "next step"],
        "keywords_2": ["recommendation", "key learning", "lessons learned", "we recommend",
                       "area for improvement", "future action"]
    },
}


def score_principle(text_lower: str, patterns: dict) -> int:
    """Score 0, 1, or 2 for a given principle."""
    # Check for substantive evidence (level 2)
    for kw in patterns["keywords_2"]:
        if kw in text_lower:
            return 2
    # Check for mention (level 1)
    count = sum(1 for kw in patterns["keywords_1"] if kw in text_lower)
    return 1 if count >= 2 else (1 if count == 1 else 0)


def classify_sector(text_lower: str, title_lower: str) -> str:
    combined = text_lower[:3000] + " " + title_lower  # Focus on beginning of text
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[sector] = score
    if scores:
        return max(scores, key=scores.get)
    return "other"


def classify_country(text_lower: str) -> str:
    scores = {}
    sample = text_lower[:5000]  # First 5k chars more likely to have location
    for country, keywords in COUNTRY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in sample)
        if score > 0:
            scores[country] = score
    if scores:
        # Bias toward UK for ties (most common)
        best = max(scores, key=scores.get)
        return best
    return "Unknown"


def extract_year(record: dict) -> int | None:
    """Extract the REPORT year (not database upload year).
    Priority: PDF text (first 1000 chars) > title keyword > discard website dates.
    published_date/scraped_at reflect when added to website, not report year.
    """
    text = record.get("pdf_text_extract", "") or ""

    # Strategy 1: Find year in first 800 chars of PDF (cover page / exec summary)
    years_found = re.findall(r'\b(20\d{2}|199\d)\b', text[:800])
    valid_years = [int(y) for y in years_found if 1996 <= int(y) <= 2025]
    if valid_years:
        # Take the MINIMUM year (earliest mention = likely report year)
        return min(valid_years)

    # Strategy 2: Year in title
    title = record.get("title", "") or ""
    m = re.search(r'\b(20\d{2}|199\d)\b', title)
    if m:
        y = int(m.group(1))
        if 1996 <= y <= 2025:
            return y

    # Strategy 3: Wider PDF search (first 3000 chars)
    years_found = re.findall(r'\b(20\d{2}|199\d)\b', text[:3000])
    valid_years = [int(y) for y in years_found if 1996 <= int(y) <= 2025]
    if valid_years:
        return min(valid_years)

    return None


def detect_assurance(text: str, title: str) -> bool:
    """Detect if a report mentions SVI/SROI Network assurance."""
    combined = (text[:3000] + " " + title).lower()
    patterns = ["assured by", "report assurance", "sroi network assured",
                "social value uk assured", "svi assured", "assurance standard",
                "has been assured", "this report has been", "assurance certificate"]
    return any(p in combined for p in patterns)


def classify_org_type(text_lower: str) -> str:
    if any(kw in text_lower[:3000] for kw in
           ["charity", "charitable", "registered charity", "ngo", "non-profit",
            "nonprofit", "not-for-profit", "voluntary organisation"]):
        return "charity"
    if any(kw in text_lower[:3000] for kw in
           ["social enterprise", "community interest company", "cic", "social business"]):
        return "social_enterprise"
    if any(kw in text_lower[:3000] for kw in
           ["council", "local authority", "government", "public sector", "nhs trust",
            "local government", "municipal"]):
        return "government"
    if any(kw in text_lower[:3000] for kw in
           ["company", "corporation", "plc", "ltd", "private sector", "business",
            "corporate", "investor"]):
        return "private"
    return "unknown"


# ─── Main processing ──────────────────────────────────────────────────────────
def main():
    records = []
    with open(INPUT) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    enriched = []
    for r in records:
        text = r.get("pdf_text_extract", "") or ""
        title = r.get("title", "") or ""
        text_lower = text.lower()
        title_lower = title.lower()

        # Metadata enrichment
        sector_clean    = classify_sector(text_lower, title_lower)
        country_clean   = classify_country(text_lower)
        year_clean      = extract_year(r)
        report_type_clean = classify_report_type(text, title)
        org_type        = classify_org_type(text_lower)
        assurance_clean = bool(r.get("is_assured")) or detect_assurance(text, title)

        # SVI quality scoring (only if we have PDF text)
        has_text = len(text) > 200
        quality_scores = {}
        for principle, patterns in PRINCIPLE_PATTERNS.items():
            if has_text:
                quality_scores[principle] = score_principle(text_lower, patterns)
            else:
                quality_scores[principle] = None  # Cannot score without text

        total_quality = (
            sum(v for v in quality_scores.values() if v is not None)
            if has_text else None
        )
        max_possible = 16  # 8 principles × max 2 points each

        # Ratio
        sroi_ratio = r.get("sroi_ratio_value")
        if sroi_ratio is None:
            sroi_ratio = r.get("sroi_ratio_pdf")

        # Stakeholder count
        stakeholder_mentions = r.get("stakeholder_mentions", 0) or 0

        row = {
            "id":               r.get("id"),
            "slug":             r.get("slug", ""),
            "title":            title[:100],
            "url":              r.get("url", ""),
            # Enriched metadata
            "sector_clean":     sector_clean,
            "country_clean":    country_clean,
            "year_clean":       year_clean,
            "report_type_clean": report_type_clean,
            "org_type":         org_type,
            "assurance_clean":  int(assurance_clean),
            # SROI metrics
            "sroi_ratio_value": sroi_ratio,
            "total_investment_raw": r.get("total_investment_raw", ""),
            "stakeholder_mentions": stakeholder_mentions,
            # PDF availability
            "has_pdf":          r.get("has_pdf", 0),
            "pdf_extracted":    r.get("pdf_extracted", 0),
            "has_text":         int(has_text),
            "text_length":      len(text),
            # Quality scores (SVI 8 principles)
            **quality_scores,
            "quality_total":    total_quality,
            "quality_pct":      round(total_quality / max_possible * 100, 1) if total_quality is not None else None,
            # Raw fields for reference
            "is_assured_raw":   r.get("is_assured", 0),
            "publication_year_raw": r.get("publication_year"),
        }
        enriched.append(row)

    # ── Write full clean dataset ───────────────────────────────────────────────
    fieldnames = list(enriched[0].keys())
    out_full = OUTDIR / "sroi_clean_dataset.csv"
    with open(out_full, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    print(f"Written: {out_full} ({len(enriched)} rows)")

    # ── Write ratios subset ───────────────────────────────────────────────────
    ratios_subset = [r for r in enriched if r["sroi_ratio_value"] is not None]
    out_ratios = OUTDIR / "sroi_ratios_subset.csv"
    with open(out_ratios, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ratios_subset)
    print(f"Written: {out_ratios} ({len(ratios_subset)} rows with SROI ratios)")

    # ── Summary statistics ─────────────────────────────────────────────────────
    from collections import Counter
    sectors   = Counter(r["sector_clean"] for r in enriched)
    countries = Counter(r["country_clean"] for r in enriched)
    years     = Counter(r["year_clean"] for r in enriched if r["year_clean"])
    rtypes    = Counter(r["report_type_clean"] for r in enriched)
    org_types = Counter(r["org_type"] for r in enriched)
    assured_count  = sum(r["assurance_clean"] for r in enriched)
    has_text_count = sum(r["has_text"] for r in enriched)

    # Quality averages (only reports with text)
    with_text = [r for r in enriched if r["has_text"] and r["quality_total"] is not None]
    if with_text:
        avg_quality = sum(r["quality_total"] for r in with_text) / len(with_text)
        avg_quality_pct = sum(r["quality_pct"] for r in with_text) / len(with_text)
        principle_avgs = {
            p: sum(r[p] for r in with_text if r[p] is not None) / len(with_text)
            for p in PRINCIPLE_PATTERNS
        }
    else:
        avg_quality = avg_quality_pct = 0
        principle_avgs = {}

    # Ratio statistics
    ratios = [r["sroi_ratio_value"] for r in enriched if r["sroi_ratio_value"] is not None]
    if ratios:
        import statistics
        ratio_stats = {
            "n": len(ratios),
            "mean": round(statistics.mean(ratios), 2),
            "median": round(statistics.median(ratios), 2),
            "stdev": round(statistics.stdev(ratios), 2),
            "min": min(ratios),
            "max": max(ratios),
        }
    else:
        ratio_stats = {}

    summary_path = OUTDIR / "enrichment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SROI DATASET ENRICHMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total reports:        {len(enriched)}\n")
        f.write(f"Reports with PDF text:{has_text_count}\n")
        f.write(f"Assured reports:      {assured_count}\n")
        f.write(f"Reports with SROI ratio: {len(ratios_subset)}\n\n")

        f.write("--- SECTOR DISTRIBUTION ---\n")
        for k, v in sectors.most_common():
            f.write(f"  {k:<25} {v}\n")

        f.write("\n--- COUNTRY DISTRIBUTION ---\n")
        for k, v in countries.most_common(15):
            f.write(f"  {k:<25} {v}\n")

        f.write("\n--- YEAR DISTRIBUTION ---\n")
        for k, v in sorted(years.items(), key=lambda x: (x[0] is None, x[0])):
            f.write(f"  {k!s:<10} {v}\n")

        f.write("\n--- REPORT TYPE ---\n")
        for k, v in rtypes.most_common():
            f.write(f"  {k:<25} {v}\n")

        f.write("\n--- ORGANIZATION TYPE ---\n")
        for k, v in org_types.most_common():
            f.write(f"  {k:<25} {v}\n")

        if ratio_stats:
            f.write("\n--- SROI RATIO STATISTICS ---\n")
            for k, v in ratio_stats.items():
                f.write(f"  {k:<10} {v}\n")

        if with_text:
            f.write(f"\n--- SVI QUALITY SCORES (n={len(with_text)} reports with text) ---\n")
            f.write(f"  Average total score:  {avg_quality:.2f}/16 ({avg_quality_pct:.1f}%)\n")
            f.write("\n  By principle (avg score 0-2):\n")
            for p, avg in sorted(principle_avgs.items(), key=lambda x: x[1]):
                label = p.replace("_", " ")
                f.write(f"  {label:<35} {avg:.2f}\n")

    print(f"\nSummary written: {summary_path}")
    print("\n--- KEY STATISTICS ---")
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
