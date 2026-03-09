# SROI Dataset Data Quality Audit Report

**Date:** 2026-03-09  
**Script:** `data_audit.py`  
**Records:** 383

---

## Executive Summary

| Field | HIGH confidence | MEDIUM | LOW | NONE/UNKNOWN |
|-------|----------------|--------|-----|--------------|
| Sector | 33 (9%) | 191 | 159 | 0 |
| Country | 58 (15%) | 220 | 26 | 79 |
| Year | 200 (52%) | 33 | 115 | 35 |
| Report type | 0 (0%) | 52 | 331 | - |
| SROI ratio | 43 VERIFIED | - | 21 UNVERIFIED | 0 UNVERIFIABLE |

---

## 1. Sector Classification

- **High confidence:** 33/383 (8.6%)
- **Medium confidence:** 191/383
- **Low confidence / tied:** 159/383

### Flagged sector classifications

| ID | Title | Flag |
|---|---|---|
| 9 | Reports Database:Whizz Kids SROI March 2011 | NARROW margin over health (score diff=1) |
| 159 | Reports Database:Social Return on Investment Analy | NARROW margin over employment (score diff=1) |
| 55 | Reports Database:Substance Abuse Prevention Dollar | NARROW margin over education (score diff=1) |
| 12 | Reports Database:Food Connect Brisbane Forecast SR | TIED between housing and sports — ambiguous |
| 83 | Reports Database:Veterans Contact Point SROI Repor | TIED between education and justice — ambiguous |
| 8 | Reports Database:HITTING THE TARGET, MISSING THE P | NARROW margin over community (score diff=1) |
| 139 | Reports Database:NSW Employment Program: Evaluatio | NARROW margin over youth (score diff=1) |
| 242 | Reports Database:Through a glass, darkly: Measurin | TIED between housing and education — ambiguous |
| 347 | Reports Database:Evaluation of Four Recovery Commu | NARROW margin over education (score diff=1) |
| 266 | Reports Database:Social Return on Investment Repor | NARROW margin over employment (score diff=1) |
| 1 | Reports Database:Subliminal Directions Social Retu | NARROW margin over education (score diff=1) |
| 53 | Reports Database:Craft Cafe SROI Evaluation | NARROW margin over housing (score diff=1) |
| 11 | Reports Database:Food Connect Sydney Forecast SROI | NARROW margin over agriculture_food (score diff=1) |
| 372 | Reports Database:The Social Value of Carmichael Ce | TIED between health and employment — ambiguous |
| 58 | Reports Database:Sunderland XL Youth Villages SROI | TIED between housing and youth — ambiguous |
| 75 | Reports Database:CUFA SROI Report | TIED between education and arts_culture — ambiguous |
| 374 | Reports Database:Community First (Moray) SROI Anal | TIED between disability and health — ambiguous |
| 142 | Reports Database:The social return on investment o | TIED between health and housing — ambiguous |
| 103 | Reports Database:Traveller Women’s Community Devel | NARROW margin over employment (score diff=1) |
| 94 | Reports Database:Ashram Employment and Skills Serv | NARROW margin over housing (score diff=1) |
| 366 | Reports Database:Cambridgeshire’s Funded Two-year- | TIED between education and elderly — ambiguous |
| 84 | Reports Database:The Social Return on Investment o | NARROW margin over None (score diff=1) |
| 7 | Reports Database:The Wise Group Cadder Environment | NARROW margin over housing (score diff=1) |
| 107 | Reports Database:An evaluation of social added val | NARROW margin over housing (score diff=1) |
| 108 | Reports Database:Care & Repair West Lothian SROI A | NARROW margin over health (score diff=1) |
| 2 | Reports Database:Perth YMCA Get Ready for Work Pro | NARROW margin over employment (score diff=1) |
| 358 | Reports Database:SROI Evaluation Analysis “Convers | NARROW margin over disability (score diff=1) |
| 100 | Reports Database:National Specialist Family Servic | NARROW margin over housing (score diff=1) |
| 296 | Reports Database:Social Return on Investment of Ta | TIED between youth and justice — ambiguous |
| 89 | Reports Database:An SROI analysis of Anchor House | TIED between housing and employment — ambiguous |

*...and 275 more — see audit_flags.csv*

---

## 2. Country Classification

- **High confidence:** 58/383
- **Unknown (even after extended search):** 79/383

### Unknown countries

| ID | Title | Flag |
|---|---|---|
| 159 | Reports Database:Social Return on Investment Analy | NARROW margin (1) |
| 55 | Reports Database:Substance Abuse Prevention Dollar | NARROW margin (1) |
| 144 | Reports Database:The YSS Pathways Accommodation Me | UNKNOWN COUNTRY — manual review needed |
| 83 | Reports Database:Veterans Contact Point SROI Repor | UNKNOWN COUNTRY — manual review needed |
| 8 | Reports Database:HITTING THE TARGET, MISSING THE P | NARROW margin (1) |
| 139 | Reports Database:NSW Employment Program: Evaluatio | NARROW margin (1) |
| 242 | Reports Database:Through a glass, darkly: Measurin | UNKNOWN COUNTRY — manual review needed |
| 131 | Reports Database:Vienna University of Economics an | UNKNOWN COUNTRY — manual review needed |
| 266 | Reports Database:Social Return on Investment Repor | UNKNOWN COUNTRY — manual review needed |
| 4 | Reports Database:North Ayrshire Fab Pad Project | NARROW margin (1) |
| 112 | Reports Database:Integrating Adaptation into REDD+ | NARROW margin (1) |
| 372 | Reports Database:The Social Value of Carmichael Ce | NARROW margin (1) |
| 58 | Reports Database:Sunderland XL Youth Villages SROI | NARROW margin (1) |
| 75 | Reports Database:CUFA SROI Report | NARROW margin (1) |
| 142 | Reports Database:The social return on investment o | UNKNOWN COUNTRY — manual review needed |
| 103 | Reports Database:Traveller Women’s Community Devel | NARROW margin (1) |
| 94 | Reports Database:Ashram Employment and Skills Serv | NARROW margin (1) |
| 366 | Reports Database:Cambridgeshire’s Funded Two-year- | UNKNOWN COUNTRY — manual review needed |
| 2 | Reports Database:Perth YMCA Get Ready for Work Pro | NARROW margin (1) |
| 358 | Reports Database:SROI Evaluation Analysis “Convers | UNKNOWN COUNTRY — manual review needed |
| 100 | Reports Database:National Specialist Family Servic | NARROW margin (1) |
| 89 | Reports Database:An SROI analysis of Anchor House | NARROW margin (1) |
| 246 | Reports Database:‘Be’ by Gentoo | UNKNOWN COUNTRY — manual review needed |
| 10 | Reports Database:RNIB and Action for Blind People  | NARROW margin (1) |
| 109 | Reports Database:Trident Reach – Homeless Link Tra | NARROW margin (1) |
| 168 | Reports Database:An evaluation of the social value | TIED UK/Australia |
| 145 | Reports Database:A Social Return on Investment ana | NARROW margin (1) |
| 20 | Reports Database:Octavia View SROI Forecast | TIED UK/USA |
| 24 | Reports Database:Country Education Foundation of A | NARROW margin (1) |
| 33 | Reports Database:Fair Gains SROI Case Study | UNKNOWN COUNTRY — manual review needed |

*...and 215 more*

---

## 3. Year Extraction

- **Explicit date pattern (HIGH):** 200/383
- **Modal year in 800 chars (MEDIUM):** 33/383
- **Low confidence:** 115/383
- **Not found:** 35/383

**Key fix from v1:** v1 used `min(valid_years)` which picked up historical reference years (1996–2001). v2 uses modal year (most frequent) and explicit date patterns, which is more robust.

### Flagged year extractions

| ID | Title | Flag |
|---|---|---|
| 55 | Reports Database:Substance Abuse Prevention Dollar | LOW CONFIDENCE — may be reference year |
| 8 | Reports Database:HITTING THE TARGET, MISSING THE P | LOW CONFIDENCE — may be reference year |
| 17 | Reports Database:Public Social Partnerships Employ | UNKNOWN YEAR — scraped date not used; set to null |
| 131 | Reports Database:Vienna University of Economics an | CHECK: only 1 occurrence |
| 75 | Reports Database:CUFA SROI Report | CHECK: only 1 occurrence |
| 374 | Reports Database:Community First (Moray) SROI Anal | LOW CONFIDENCE — may be reference year |
| 32 | Reports Database:Measuring the SROI of Stage 3 Ada | UNKNOWN YEAR — scraped date not used; set to null |
| 119 | Reports Database:Choices Plus: An SROI Study | CHECK: only 1 occurrence |
| 358 | Reports Database:SROI Evaluation Analysis “Convers | LOW CONFIDENCE — may be reference year |
| 100 | Reports Database:National Specialist Family Servic | UNKNOWN YEAR — scraped date not used; set to null |
| 33 | Reports Database:Fair Gains SROI Case Study | LOW CONFIDENCE — may be reference year |
| 28 | Reports Database:Future Jobs Fund Programme SROI E | LOW CONFIDENCE — may be reference year |
| 172 | Reports Database:Furnish SROI Report | LOW CONFIDENCE — may be reference year |
| 39 | Reports Database:Hidden Value: Demonstrating the e | LOW CONFIDENCE — may be reference year |
| 251 | Reports Database:The Social Return On Investment f | CHECK: only 1 occurrence |
| 169 | Reports Database:Striking the right balance: a soc | LOW CONFIDENCE — may be reference year |
| 5 | Reports Database:Investing for Social Value | UNKNOWN YEAR — scraped date not used; set to null |
| 3 | Reports Database:Lawnmowers Independent Theatre Co | CHECK: only 1 occurrence |
| 6 | Reports Database:Dame Kelly Holmes Trust Impact Re | CHECK: only 1 occurrence |
| 16 | Reports Database:Pachamama Coffee Co-op | LOW CONFIDENCE — may be reference year |

---

## 4. Report Type Classification

- **Evaluative:** 339 (322 assumed, 17 keyword-confirmed)
- **Forecast:** 40
- **Scoping:** 4

**Key fix from v1:** v1 coded reports without type keywords as 'Unknown'. v2 codes them as 'Evaluative' with `type_note='assumed_evaluative'` and `type_confidence='LOW'`. This is documented explicitly. Analysts can filter to keyword-confirmed types only using `type_note != 'assumed_evaluative'`.


---

## 5. SROI Ratio Validation

- **Total reports with ratio:** 64/383
- **VERIFIED (found in PDF text):** 43
- **UNVERIFIED (in metadata but not PDF text):** 21
- **UNVERIFIABLE (no PDF text):** 0

### Unverified ratios

| ID | Title | Flag |
|---|---|---|
| 159 | Reports Database:Social Return on Investment Analy | Ratio 27.0 NOT FOUND in PDF text — may be from metadata only |
| 83 | Reports Database:Veterans Contact Point SROI Repor | Ratio 15.7 NOT FOUND in PDF text — may be from metadata only |
| 8 | Reports Database:HITTING THE TARGET, MISSING THE P | Ratio 14.0 NOT FOUND in PDF text — may be from metadata only |
| 347 | Reports Database:Evaluation of Four Recovery Commu | Ratio 9.24 NOT FOUND in PDF text — may be from metadata only |
| 53 | Reports Database:Craft Cafe SROI Evaluation | Ratio 8.27 NOT FOUND in PDF text — may be from metadata only |
| 32 | Reports Database:Measuring the SROI of Stage 3 Ada | Ratio 6.0 NOT FOUND in PDF text — may be from metadata only |
| 103 | Reports Database:Traveller Women’s Community Devel | Ratio 5.6 NOT FOUND in PDF text — may be from metadata only |
| 94 | Reports Database:Ashram Employment and Skills Serv | Ratio 5.19 NOT FOUND in PDF text — may be from metadata only |
| 84 | Reports Database:The Social Return on Investment o | Ratio 4.7 NOT FOUND in PDF text — may be from metadata only |
| 358 | Reports Database:SROI Evaluation Analysis “Convers | Ratio 4.15 NOT FOUND in PDF text — may be from metadata only |
| 100 | Reports Database:National Specialist Family Servic | Ratio 3.95 NOT FOUND in PDF text — may be from metadata only |
| 296 | Reports Database:Social Return on Investment of Ta | Ratio 3.5 NOT FOUND in PDF text — may be from metadata only |
| 246 | Reports Database:‘Be’ by Gentoo | Ratio 3.35 NOT FOUND in PDF text — may be from metadata only |
| 168 | Reports Database:An evaluation of the social value | Ratio 3.2 NOT FOUND in PDF text — may be from metadata only |
| 145 | Reports Database:A Social Return on Investment ana | Ratio 3.19 NOT FOUND in PDF text — may be from metadata only |
| 24 | Reports Database:Country Education Foundation of A | Ratio 3.1 NOT FOUND in PDF text — may be from metadata only |
| 28 | Reports Database:Future Jobs Fund Programme SROI E | Ratio 2.9 NOT FOUND in PDF text — may be from metadata only |
| 172 | Reports Database:Furnish SROI Report | Ratio 2.7 NOT FOUND in PDF text — may be from metadata only |
| 99 | Reports Database:Social Impact Analysis for Sewa D | Ratio 2.6 NOT FOUND in PDF text — may be from metadata only |
| 268 | Reports Database:The economic and social impact of | Ratio 1.9 NOT FOUND in PDF text — may be from metadata only |
| 95 | Reports Database:Nottingham City Homes; Decent Hom | Ratio 1.58 NOT FOUND in PDF text — may be from metadata only |

---

## 6. Quality Score Audit

### p1_involve_stakeholders
- Score 0: 96 | Score 1: 144 | Score 2: 136
- Mean: 1.106
- **WEAK flags (79 reports):** single keyword hit — review manually for false positives

### p2_understand_changes
- Score 0: 6 | Score 1: 259 | Score 2: 111
- Mean: 1.279
- **WEAK flags (22 reports):** single keyword hit — review manually for false positives

### p3_value_what_matters
- Score 0: 98 | Score 1: 235 | Score 2: 43
- Mean: 0.854
- **WEAK flags (202 reports):** single keyword hit — review manually for false positives

### p4_only_material
- Score 0: 223 | Score 1: 124 | Score 2: 29
- Mean: 0.484
- **WEAK flags (98 reports):** single keyword hit — review manually for false positives

### p5_do_not_overclaim
- Score 0: 319 | Score 1: 10 | Score 2: 47
- Mean: 0.277
- **WEAK flags (10 reports):** single keyword hit — review manually for false positives

### p6_be_transparent
- Score 0: 69 | Score 1: 200 | Score 2: 107
- Mean: 1.101
- **WEAK flags (122 reports):** single keyword hit — review manually for false positives

### p7_verify_result
- Score 0: 175 | Score 1: 138 | Score 2: 63
- Mean: 0.702
- **WEAK flags (120 reports):** single keyword hit — review manually for false positives

### p8_be_responsive
- Score 0: 172 | Score 1: 117 | Score 2: 87
- Mean: 0.774
- **WEAK flags (90 reports):** single keyword hit — review manually for false positives

---

## 7. Overall Data Quality Assessment

| Dimension | Assessment |
|-----------|------------|
| Sector classification | 9% HIGH confidence; 159 tied/ambiguous |
| Country classification | 79 reports (after extended search) remain Unknown |
| Year extraction | Fixed: modal vs. min strategy reduces error rate |
| Report type | 0 keyword-confirmed; 331 assumed Evaluative |
| SROI ratios | 43/64 verified in PDF text; 21 metadata-only |
| Quality scores | Pattern-based; see classification_evidence.csv for per-report evidence |

**Recommendation:** Use `sector_confidence`, `country_confidence`, `year_confidence`, and `type_confidence` columns to filter analyses to higher-confidence subsets in robustness checks.
