"""
Round 5 — empirical IRC:82 compliance audit.

Reads an eval JSON (per_image_results) and measures how well the model
actually adheres to the IRC:82-2015 taxonomy + severity criteria when
applied to real images.

Metrics produced:
  - IRC label compliance rate     (% of emitted labels that are canonical IRC names)
  - Legacy-label slip-through rate (% emitted labels needing canonicalization)
  - Hallucination rate            (% emitted labels NOT in the 18 IRC types)
  - IRC section citation rate     (% of descriptions citing "§X.Y.Z" or "IRC:82")
  - Severity-format compliance    (% of severities using valid IRC tier names)
  - Severity-N/A handling         (% of severity-not-applicable types emitting N/A)

Usage:
  python scripts/analyze_irc_compliance.py eval_results/attain_irc_audit_100.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.irc82_taxonomy import (
    IRC82_DISTRESS_TAXONOMY,
    canonicalize_to_irc,
    is_irc_valid,
)


VALID_IRC_NAMES = set(IRC82_DISTRESS_TAXONOMY.keys())
VALID_SEVERITY_TIERS = {"Low", "Medium", "High", "Small", "Large", "N/A", "None"}
TYPES_WITH_NO_SEVERITY = {
    name for name, e in IRC82_DISTRESS_TAXONOMY.items()
    if e.get("severity") is None
}
TYPES_WITH_SMALL_MEDIUM_LARGE = {"Potholes"}  # IRC §7.5.3.4 uses Small/Medium/Large


def analyze(path: Path) -> dict:
    """Run the full Round 5 compliance audit on an eval JSON."""
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_image_results") or []

    # ============================================================
    # Pass 1 — collect raw emissions
    # ============================================================
    raw_label_emissions: list[str] = []
    canonicalized_emissions: list[str] = []
    severities: list[str] = []
    descriptions: list[str] = []
    stage2_raws: list[str] = []
    for r in rows:
        stage2_raw = r.get("stage2_raw") or ""
        stage2_raws.append(stage2_raw)
        # Extract the raw DISTRESS_TYPES line from stage2_raw if available
        if stage2_raw:
            m = re.search(r"DISTRESS_TYPES:\s*([^\n]+)", stage2_raw, re.IGNORECASE)
            if m:
                raw = [t.strip() for t in m.group(1).split(",") if t.strip()]
                raw_label_emissions.extend(raw)
        # Already-canonicalized pred_pipeline
        for t in (r.get("pred_pipeline") or []):
            canonicalized_emissions.append(t)
        sv = r.get("pred_severity") or ""
        if sv:
            severities.append(sv.strip())
        # Extract description for citation check
        if stage2_raw:
            m = re.search(r"DESCRIPTION:\s*(.+)", stage2_raw, re.IGNORECASE | re.DOTALL)
            if m:
                descriptions.append(m.group(1).strip())

    # ============================================================
    # Metric 1 — IRC label compliance rate
    # ============================================================
    # Of all raw emissions (what the model literally said), how many are
    # already canonical IRC names without needing the canonicalizer?
    raw_irc_compliant = sum(1 for t in raw_label_emissions if t in VALID_IRC_NAMES)
    irc_compliance_rate = raw_irc_compliant / max(len(raw_label_emissions), 1)

    # ============================================================
    # Metric 2 — Legacy slip-through (model said legacy, canonicalizer fixed it)
    # ============================================================
    legacy_slipped = 0
    for raw in raw_label_emissions:
        if raw in VALID_IRC_NAMES:
            continue  # already IRC, not legacy
        canonical = canonicalize_to_irc(raw)
        if canonical and canonical in VALID_IRC_NAMES:
            legacy_slipped += 1  # was non-IRC, canonicalizer rescued
    legacy_slip_rate = legacy_slipped / max(len(raw_label_emissions), 1)

    # ============================================================
    # Metric 3 — Hallucination rate (raw label that doesn't map to any IRC)
    # ============================================================
    hallucinations: Counter = Counter()
    for raw in raw_label_emissions:
        if raw in VALID_IRC_NAMES:
            continue
        canonical = canonicalize_to_irc(raw)
        if canonical is None:
            # Was filtered (Other/Unknown family) — count separately
            continue
        if canonical not in VALID_IRC_NAMES:
            hallucinations[raw] += 1
    hallucination_rate = sum(hallucinations.values()) / max(len(raw_label_emissions), 1)

    # ============================================================
    # Metric 4 — IRC section citation rate in descriptions
    # ============================================================
    cited = 0
    for desc in descriptions:
        if re.search(r"IRC:?82|§\s*7\.\d", desc):
            cited += 1
    citation_rate = cited / max(len(descriptions), 1)

    # ============================================================
    # Metric 5 — Severity format compliance
    # Computed only over images where the model actually predicted distress
    # (severity is meaningless when Stage 1 = Normal). The eval script
    # over-writes Normal-case severity to "Unknown" which is a downstream
    # normalization quirk, not a real compliance failure.
    # ============================================================
    sev_tier_dist = Counter(severities)
    # Severity over distressed-only predictions (where severity actually matters)
    distressed_severities = [
        (r.get("pred_severity") or "").strip()
        for r in rows
        if r.get("is_distressed_pred")
    ]
    distressed_sev_dist = Counter(distressed_severities)
    valid_sev_distressed = sum(
        v for k, v in distressed_sev_dist.items() if k in VALID_SEVERITY_TIERS
    )
    sev_compliance = valid_sev_distressed / max(sum(distressed_sev_dist.values()), 1)

    # ============================================================
    # Metric 6 — N/A handling — for types declared severity-not-applicable,
    # is the model correctly outputting N/A or omitting severity?
    # ============================================================
    na_handling = {"correct_na": 0, "wrong_concrete_severity": 0, "total": 0}
    for r in rows:
        preds = r.get("pred_pipeline") or []
        sev = (r.get("pred_severity") or "").strip()
        # Did the model emit ANY type whose IRC severity is not-applicable?
        emitted_na_types = [t for t in preds if t in TYPES_WITH_NO_SEVERITY]
        if not emitted_na_types:
            continue
        na_handling["total"] += 1
        # If ALL emitted types are N/A-only, severity should be N/A or None
        if all(t in TYPES_WITH_NO_SEVERITY for t in preds):
            if sev in ("N/A", "None", "n/a", ""):
                na_handling["correct_na"] += 1
            else:
                na_handling["wrong_concrete_severity"] += 1
        # If MIXED (e.g. has Potholes + Hungry Surface), concrete severity is correct
        # because IRC rates by the worst applicable distress.

    # ============================================================
    # Metric 7 — Pothole severity uses Small/Medium/Large not Low/Medium/High
    # ============================================================
    pothole_sev_compliance = {"correct_terminology": 0, "wrong_lmh": 0, "total": 0}
    for r in rows:
        preds = r.get("pred_pipeline") or []
        sev = (r.get("pred_severity") or "").strip()
        if "Potholes" not in preds:
            continue
        # Only check if Pothole is the ONLY emitted type (otherwise severity is
        # ambiguous across multiple distresses)
        if preds != ["Potholes"]:
            continue
        pothole_sev_compliance["total"] += 1
        if sev in ("Small", "Medium", "Large"):
            pothole_sev_compliance["correct_terminology"] += 1
        elif sev in ("Low", "High"):
            pothole_sev_compliance["wrong_lmh"] += 1

    # ============================================================
    # Distribution stats
    # ============================================================
    label_dist = Counter(canonicalized_emissions)
    # Multi-class emission rate
    multi_pred = sum(1 for r in rows if len(r.get("pred_pipeline") or []) >= 2)

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "n_images": len(rows),
        "n_label_emissions_total": len(raw_label_emissions),
        "irc_label_compliance_rate": round(irc_compliance_rate, 4),
        "legacy_slip_through_rate": round(legacy_slip_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "irc_section_citation_rate": round(citation_rate, 4),
        "severity_format_compliance_distressed_only": round(sev_compliance, 4),
        "distressed_severity_distribution": dict(distressed_sev_dist.most_common()),
        "na_severity_handling": na_handling,
        "pothole_severity_terminology": pothole_sev_compliance,
        "label_distribution": dict(label_dist.most_common()),
        "severity_distribution": dict(sev_tier_dist.most_common()),
        "hallucinations": dict(hallucinations.most_common(10)),
        "multi_class_prediction_rate": round(multi_pred / max(len(rows), 1), 4),
        "n_descriptions_examined": len(descriptions),
    }


def render_markdown(result: dict) -> str:
    L = []
    L.append("# Round 5 — Empirical IRC:82 Compliance Audit")
    L.append("")
    L.append(f"Source: `{result['source_file']}`  ({result['n_images']} images, "
             f"{result['n_label_emissions_total']} total label emissions)")
    L.append("")
    L.append("## Headline compliance metrics")
    L.append("")
    L.append("| Metric | Rate | Interpretation |")
    L.append("|---|---|---|")
    L.append(f"| **IRC label compliance** (raw model output is exactly a canonical IRC name) "
             f"| **{result['irc_label_compliance_rate']:.1%}** "
             f"| Higher = model better follows the new vocabulary |")
    L.append(f"| **Legacy slip-through** (model said RDD-style code, canonicalizer fixed it) "
             f"| {result['legacy_slip_through_rate']:.1%} "
             f"| Lower = prompts are working; high = adapter influence |")
    L.append(f"| **Hallucination** (model invented label not in any taxonomy) "
             f"| {result['hallucination_rate']:.1%} "
             f"| Lower is better; should be < 5% |")
    L.append(f"| **IRC section citation** (description cites '§7.X.Y' or 'IRC:82') "
             f"| **{result['irc_section_citation_rate']:.1%}** "
             f"| Higher = report follows IRC-format expectations |")
    L.append(f"| **Severity format compliance** (over distressed predictions) "
             f"| **{result['severity_format_compliance_distressed_only']:.1%}** "
             f"| Should be ~100% |")
    L.append(f"| **Multi-class prediction rate** (image gets 2+ distress labels) "
             f"| {result['multi_class_prediction_rate']:.1%} "
             f"| IRC §7.1 explicitly expects multi-class on damaged roads |")
    L.append("")

    L.append("## Severity handling for IRC 'not-applicable' types")
    L.append("")
    na = result["na_severity_handling"]
    if na["total"] > 0:
        L.append(f"On {na['total']} images where the model emitted only IRC types "
                 f"with severity declared not-applicable (Streaking, Hungry Surface, "
                 f"Hairline, Corrugation, Shoving, Shallow Depression, Settlement, "
                 f"Stripping, Edge Breaking):")
        L.append("")
        L.append(f"- Correctly emitted N/A or None: {na['correct_na']} / {na['total']} "
                 f"({na['correct_na']/na['total']:.0%})")
        L.append(f"- Wrongly emitted a concrete L/M/H tier: {na['wrong_concrete_severity']} "
                 f"/ {na['total']} ({na['wrong_concrete_severity']/na['total']:.0%})")
    else:
        L.append("No qualifying images in this sample (no images had ONLY severity-N/A IRC types).")
    L.append("")

    L.append("## Pothole-specific severity terminology")
    L.append("")
    ps = result["pothole_severity_terminology"]
    if ps["total"] > 0:
        L.append(f"On {ps['total']} images where Potholes is the ONLY emitted distress "
                 f"(IRC §7.5.3.4 specifies Small/Medium/Large, NOT Low/Medium/High):")
        L.append("")
        L.append(f"- Correctly used Small/Medium/Large: {ps['correct_terminology']} / "
                 f"{ps['total']} ({ps['correct_terminology']/ps['total']:.0%})")
        L.append(f"- Used L/M/H instead (wrong terminology, still recognizable): "
                 f"{ps['wrong_lmh']} / {ps['total']} ({ps['wrong_lmh']/ps['total']:.0%})")
    else:
        L.append("No qualifying images in this sample.")
    L.append("")

    L.append("## Label distribution (canonicalized)")
    L.append("")
    L.append("| IRC distress type | Emissions | Tier-1 in our 18? |")
    L.append("|---|---|---|")
    for lbl, n in result["label_distribution"].items():
        in_taxonomy = "✓" if lbl in IRC82_DISTRESS_TAXONOMY else "✗"
        L.append(f"| {lbl} | {n} | {in_taxonomy} |")
    L.append("")

    L.append("## Severity tier distribution")
    L.append("")
    L.append("| Severity | Emissions |")
    L.append("|---|---|")
    for sev, n in result["severity_distribution"].items():
        L.append(f"| `{sev}` | {n} |")
    L.append("")

    if result["hallucinations"]:
        L.append("## Hallucinated labels (model invented these — not in any taxonomy)")
        L.append("")
        L.append("| Hallucinated label | Count |")
        L.append("|---|---|")
        for lbl, n in result["hallucinations"].items():
            L.append(f"| `{lbl}` | {n} |")
    else:
        L.append("## Hallucinations")
        L.append("")
        L.append("**None.** Every model emission either matched a canonical IRC name or "
                 "was a known legacy alias the canonicalizer handled.")
    L.append("")

    return "\n".join(L)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_irc_compliance.py <eval_results/foo.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result = analyze(path)
    out_json = PROJECT_ROOT / "eval_results" / "irc82_compliance_audit.json"
    out_md = PROJECT_ROOT / "eval_results" / "irc82_compliance_audit.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")

    print("=" * 78)
    print("Round 5 — Empirical IRC:82 Compliance Audit")
    print("=" * 78)
    print()
    print(f"Sample: {result['n_images']} images, "
          f"{result['n_label_emissions_total']} total label emissions")
    print()
    print(f"  IRC label compliance:      {result['irc_label_compliance_rate']:.1%}")
    print(f"  Legacy slip-through rate:  {result['legacy_slip_through_rate']:.1%}")
    print(f"  Hallucination rate:        {result['hallucination_rate']:.1%}")
    print(f"  IRC section citation rate: {result['irc_section_citation_rate']:.1%}")
    print(f"  Severity format compliance (distressed only): {result['severity_format_compliance_distressed_only']:.1%}")
    print(f"  Multi-class prediction:    {result['multi_class_prediction_rate']:.1%}")
    print()
    print(f"  N/A handling:    {result['na_severity_handling']}")
    print(f"  Pothole terms:   {result['pothole_severity_terminology']}")
    print()
    if result["hallucinations"]:
        print(f"  Hallucinations: {result['hallucinations']}")
    else:
        print("  Hallucinations: NONE — all emissions covered by IRC taxonomy + aliases")
    print()
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")


if __name__ == "__main__":
    main()
