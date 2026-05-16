# Round 5 — Empirical IRC:82 Compliance Audit

Source: `eval_results\attain_irc_audit_100.json`  (100 images, 145 total label emissions)

## Headline compliance metrics

| Metric | Rate | Interpretation |
|---|---|---|
| **IRC label compliance** (raw model output is exactly a canonical IRC name) | **100.0%** | Higher = model better follows the new vocabulary |
| **Legacy slip-through** (model said RDD-style code, canonicalizer fixed it) | 0.0% | Lower = prompts are working; high = adapter influence |
| **Hallucination** (model invented label not in any taxonomy) | 0.0% | Lower is better; should be < 5% |
| **IRC section citation** (description cites '§7.X.Y' or 'IRC:82') | **100.0%** | Higher = report follows IRC-format expectations |
| **Severity format compliance** (over distressed predictions) | **100.0%** | Should be ~100% |
| **Multi-class prediction rate** (image gets 2+ distress labels) | 69.0% | IRC §7.1 explicitly expects multi-class on damaged roads |

## Severity handling for IRC 'not-applicable' types

No qualifying images in this sample (no images had ONLY severity-N/A IRC types).

## Pothole-specific severity terminology

No qualifying images in this sample.

## Label distribution (canonicalized)

| IRC distress type | Emissions | Tier-1 in our 18? |
|---|---|---|
| Longitudinal Cracking | 69 | ✓ |
| Transverse Cracking | 42 | ✓ |
| Potholes | 25 | ✓ |
| Alligator Cracking | 8 | ✓ |
| Bleeding | 1 | ✓ |

## Severity tier distribution

| Severity | Emissions |
|---|---|
| `Medium` | 62 |
| `Unknown` | 31 |
| `Low` | 7 |

## Hallucinations

**None.** Every model emission either matched a canonical IRC name or was a known legacy alias the canonicalizer handled.
