# IRC:82-2015 Robustness Verification Report

Generated: 2026-05-16T04:59:50.899266Z

Five independent verification rounds. Synthetic rounds (1-4) check 
the static implementation against the IRC:82-2015 PDF and against 
a 32-case parser suite. The empirical round (5) audits real model 
output on Attain images to verify the prompts produce IRC-compliant 
output in production.

## Summary scoreboard

| Round | Test | Status | Score |
|---|---|---|---|
| 1 | PDF cross-reference — every §7 subsection accounted for | **PASS** | 18 mapped + 2 excluded = 20 expected |
| 2 | Parser canonicalization — 32 synthetic edge cases | **PASS** | 32/32 cases passed |
| 3 | Severity criteria match — mm thresholds in PDF text | **PASS** | 16/16 threshold strings found in PDF |
| 4 | Prompt content audit — IRC refs, forbidden labels, all 18 types | **PASS** | v1 + v2 both compliant (18 IRC types named in each) |
| 5 | Empirical compliance — model behavior on real Attain images | see below | 100 images, 145 emissions |

## Round 1 — PDF cross-reference detail

Every numbered subsection in IRC:82-2015 Section 7 (the distress taxonomy section) 
must be either mapped to a canonical name in our taxonomy or explicitly excluded 
with a documented reason.

| §7 subsection | Status | Maps to / reason |
|---|---|---|
| §7.2.1 | ✓ ok-mapped | Bleeding |
| §7.2.2 | ✓ ok-excluded | EXCLUDED:Smooth Surface (severity needs skid number measurement) |
| §7.2.3 | ✓ ok-mapped | Streaking |
| §7.2.4 | ✓ ok-mapped | Hungry Surface |
| §7.3.2 | ✓ ok-mapped | Hairline Cracks |
| §7.3.3 | ✓ ok-mapped | Alligator Cracking |
| §7.3.4 | ✓ ok-mapped | Longitudinal Cracking |
| §7.3.5 | ✓ ok-mapped | Transverse Cracking |
| §7.3.6 | ✓ ok-mapped | Edge Cracking |
| §7.3.7 | ✓ ok-excluded | EXCLUDED:Reflection Cracking (requires underlying-layer knowledge) |
| §7.4.1 | ✓ ok-mapped | Slippage |
| §7.4.2 | ✓ ok-mapped | Rutting |
| §7.4.3 | ✓ ok-mapped | Corrugation |
| §7.4.4 | ✓ ok-mapped | Shoving |
| §7.4.5 | ✓ ok-mapped | Shallow Depression |
| §7.4.6 | ✓ ok-mapped | Settlement |
| §7.5.1 | ✓ ok-mapped | Stripping |
| §7.5.2 | ✓ ok-mapped | Ravelling |
| §7.5.3 | ✓ ok-mapped | Potholes |
| §7.5.4 | ✓ ok-mapped | Edge Breaking |

**Extra '7.X.Y' patterns matched in PDF text but not in our taxonomy:**

| Pattern | Why it's not a missing distress type |
|---|---|
| §7.3.1 | Section header '7.3.1 General' (intro to §7.3 cracks, not a distress type) |
| §7.4.7 | Photo reference 'Photo 7.4.7 Upheaval' (matched by section-number regex) |

These are NOT missing distress types — they are non-distress text (section intros, photo captions) that match the regex pattern but don't describe a pavement defect.

## Round 2 — Parser canonicalization detail

Suite of 32 edge cases covering: exact IRC names, legacy RDD codes 
(D00/D10/D20/D40/D43/D44/D50), case variations, whitespace, hallucinated D-codes, 
Other/Unknown filtering, Normal detection, free-text keyword fallback, empty input.

**Result: 32/32 cases passed.**

Comparison is set-based — order of distress_types doesn't affect downstream 
metrics (TP/FP/FN use set semantics).

## Round 3 — Severity criteria match against PDF text

For each IRC distress type with quantitative severity criteria, the mm thresholds 
we encoded must appear (in normalized form) in the IRC:82-2015 PDF text.

**Result: 16/16 threshold strings matched.**

| Distress type | Tier check | Found in PDF |
|---|---|---|
| Longitudinal Cracking | Low width 1-3 mm | ✓ |
| Longitudinal Cracking | Medium width 3-6 mm | ✓ |
| Longitudinal Cracking | High width > 6 mm | ✓ |
| Transverse Cracking | Low width 1-3 mm | ✓ |
| Transverse Cracking | Medium width 3-6 mm | ✓ |
| Transverse Cracking | High width > 6 mm | ✓ |
| Alligator Cracking | Low 1-3 mm | ✓ |
| Alligator Cracking | Medium 3-6 mm | ✓ |
| Alligator Cracking | High > 6 mm | ✓ |
| Rutting | Low 4-10 mm | ✓ |
| Rutting | High > 10 mm | ✓ |
| Potholes | Small 25 mm 200 mm | ✓ |
| Potholes | Medium 25-50 mm 500 mm | ✓ |
| Potholes | Large > 50 mm > 500 mm | ✓ |
| Edge Cracking | Medium loss up to 10% | ✓ |
| Edge Cracking | High loss > 10% | ✓ |

## Round 4 — Prompt content audit

Both Stage 2 system prompts (v1 production + v2 Improved Baseline) must:
- Contain the literal string `IRC:82-2015`
- Forbid non-IRC labels in a negative-instruction context
- Name all 18 IRC vision-visible distress types verbatim
- Cite IRC section numbers (§)
- Contain quantitative severity criteria (mm thresholds)

| Prompt | Status | IRC marker | Forbids non-IRC | Types named | Sections cited | Severity |
|---|---|---|---|---|---|---|
| v1_stage2 | **PASS** | ✓ | ✓ | 18/18 | ✓ | ✓ |
| v2_stage2 | **PASS** | ✓ | ✓ | 18/18 | ✓ | ✓ |

## Round 5 — Empirical compliance on real Attain images

Sample: 100 images from Attain WS_V2.0, processed by 
the base Qwen2.5-VL-7B-Instruct model with v2 Improved Baseline IRC prompts 
(no adapter). All emissions audited for IRC compliance.

| Metric | Rate | Interpretation |
|---|---|---|
| **Raw IRC label compliance** (model output is exactly canonical IRC name) | **100.0%** | Higher = prompts working |
| **Legacy slip-through** (canonicalizer needed to fix model output) | 0.0% | Lower is better |
| **Hallucination** (label not in any taxonomy) | 0.0% | Should be < 5% |
| **IRC section citation** (description cites § or 'IRC:82') | **100.0%** | |
| **Severity format compliance** (distressed predictions only) | 100.0% | Computed over images where the model predicted distress |
| **Multi-class prediction rate** | 69.0% | IRC §7.1 expects multi-class |

**Hallucinations: NONE.** Every model emission either was a canonical IRC name or a known legacy alias the canonicalizer normalized.

### Label distribution (canonicalized to IRC)

| IRC distress type | Emissions |
|---|---|
| Longitudinal Cracking | 69 |
| Transverse Cracking | 42 |
| Potholes | 25 |
| Alligator Cracking | 8 |
| Bleeding | 1 |

## Conclusion

**Synthetic rounds (1–4): 4/4 pass.**

**Empirical round (5):** model emits IRC-canonical labels 100% of the time. 
Hallucination rate is 0.0% (< 5% threshold). Section-citation rate is 100%.

The IRC:82-2015 alignment is verified at the data layer (taxonomy + parser), 
the prompt layer (v1 + v2 both compliant), and the runtime layer (empirical 
model output on real images). Legacy data remains backward-compatible through 
`canonicalize_to_irc()` running on parse.