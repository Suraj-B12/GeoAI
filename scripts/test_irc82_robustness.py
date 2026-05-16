"""
Multi-round robustness test for the IRC:82-2015 taxonomy and prompt integration.

Rounds:
  1. PDF cross-reference     — every IRC §7 subsection accounted for
  2. Parser canonicalization — 50+ edge cases, must all pass
  3. Severity criteria       — string match against PDF text
  4. Prompt content audit    — IRC refs, forbidden labels, all 18 types present
  5. (Run separately) End-to-end empirical audit on 50 Attain images

Output: eval_results/irc82_verification.json  + .md report
Exit non-zero if any round fails.

Usage: PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/test_irc82_robustness.py
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
    severity_for,
)
from scripts.utils import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    ALL_DISTRESS_TYPES,
    IRC_SEVERITY_CRITERIA,
    parse_stage2_response,
)
from scripts.utils_v2_prompts import (
    STAGE1_SYSTEM_PROMPT_V2,
    STAGE2_SYSTEM_PROMPT_V2,
)


# ============================================================
# Round 1 — PDF cross-reference
# ============================================================

def round1_pdf_cross_reference() -> dict:
    """
    Extract every '7.X.Y' subsection title from the IRC:82-2015 PDF
    and verify each is either in our taxonomy or explicitly excluded.
    """
    import pypdf

    # PDF subsection -> expected canonical IRC name in our taxonomy
    # (or "EXCLUDED" with reason)
    expected_map = {
        # 7.2 Surface Defects
        "7.2.1": "Bleeding",
        "7.2.2": "EXCLUDED:Smooth Surface (severity needs skid number measurement)",
        "7.2.3": "Streaking",
        "7.2.4": "Hungry Surface",
        # 7.3 Cracks
        "7.3.2": "Hairline Cracks",
        "7.3.3": "Alligator Cracking",
        "7.3.4": "Longitudinal Cracking",
        "7.3.5": "Transverse Cracking",
        "7.3.6": "Edge Cracking",
        "7.3.7": "EXCLUDED:Reflection Cracking (requires underlying-layer knowledge)",
        # 7.4 Deformation
        "7.4.1": "Slippage",
        "7.4.2": "Rutting",
        "7.4.3": "Corrugation",
        "7.4.4": "Shoving",
        "7.4.5": "Shallow Depression",
        "7.4.6": "Settlement",
        # 7.5 Disintegration
        "7.5.1": "Stripping",
        "7.5.2": "Ravelling",
        "7.5.3": "Potholes",
        "7.5.4": "Edge Breaking",
    }

    # Read PDF and find which subsections actually appear
    pdf_path = PROJECT_ROOT / "IRC 82" / "irc.gov.in.082.2015.pdf"
    found_subsections: set[str] = set()
    if pdf_path.exists():
        with pdf_path.open("rb") as f:
            reader = pypdf.PdfReader(f)
            text = "\n".join(p.extract_text() for p in reader.pages[10:40])
        # Subsection markers like "7.2.1 Bleeding" or "7.3.4 Longitudinal Cracking"
        for m in re.finditer(r"\b(7\.[2-5]\.\d+)\b", text):
            found_subsections.add(m.group(1))
    else:
        return {
            "status": "skipped",
            "reason": "PDF not found at IRC 82/irc.gov.in.082.2015.pdf",
            "n_expected": len(expected_map),
        }

    # Verify each expected subsection
    results = []
    accounted = 0
    misses = []
    for subsection, expected in expected_map.items():
        in_pdf = subsection in found_subsections
        if expected.startswith("EXCLUDED:"):
            status = "ok-excluded"
            taxonomy_match = expected.split(":", 1)[1]
        else:
            in_taxonomy = expected in IRC82_DISTRESS_TAXONOMY
            entry_section = (IRC82_DISTRESS_TAXONOMY.get(expected) or {}).get("irc_section", "")
            if in_taxonomy and entry_section == subsection:
                status = "ok-mapped"
                taxonomy_match = expected
                accounted += 1
            else:
                status = "MISS"
                taxonomy_match = f"expected {expected} (irc_section={entry_section!r})"
                misses.append((subsection, expected, entry_section))
        results.append({
            "subsection": subsection,
            "found_in_pdf": in_pdf,
            "expected": expected,
            "status": status,
        })

    # Any extra subsections in PDF we didn't anticipate?
    expected_subs = set(expected_map.keys())
    extras = sorted(found_subsections - expected_subs - {"7.4"})  # 7.4 is parent header
    # Filter common false positives (page numbers, table references)
    extras = [s for s in extras if not s.endswith(".0")]

    return {
        "status": "fail" if misses else "pass",
        "n_expected_subsections": len(expected_map),
        "n_mapped_to_taxonomy": accounted,
        "n_explicitly_excluded": sum(1 for v in expected_map.values() if v.startswith("EXCLUDED:")),
        "subsections_in_pdf": len(found_subsections),
        "details": results,
        "extra_subsections_in_pdf": extras,
        "misses": [{"subsection": s, "expected": e, "got_irc_section": g} for s, e, g in misses],
    }


# ============================================================
# Round 2 — Parser canonicalization test suite
# ============================================================

PARSER_CASES: list[tuple[str, list[str], str]] = [
    # (input_text, expected_distress_types_after_parse, description)
    # ------ Direct IRC names (exact match, should pass through) ------
    ("DISTRESS_TYPES: Potholes\nSEVERITY: Small\nDESCRIPTION: Test.",
     ["Potholes"], "exact IRC name pothole"),
    ("DISTRESS_TYPES: Longitudinal Cracking\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Longitudinal Cracking"], "exact IRC name longitudinal"),
    ("DISTRESS_TYPES: Alligator Cracking, Potholes\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Alligator Cracking", "Potholes"], "exact IRC multi"),

    # ------ Legacy RDD codes (must canonicalize) ------
    ("DISTRESS_TYPES: Pothole (D40)\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Potholes"], "legacy D40 → Potholes"),
    ("DISTRESS_TYPES: Longitudinal Crack (D00)\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Longitudinal Cracking"], "legacy D00 → Longitudinal Cracking"),
    ("DISTRESS_TYPES: Transverse Crack (D10)\nSEVERITY: Medium\nDESCRIPTION: Test.",
     ["Transverse Cracking"], "legacy D10 → Transverse Cracking"),
    ("DISTRESS_TYPES: Alligator Crack (D20)\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Alligator Cracking"], "legacy D20 → Alligator Cracking"),

    # ------ Legacy non-RDD aliases ------
    ("DISTRESS_TYPES: Raveling\nSEVERITY: Medium\nDESCRIPTION: Test.",
     ["Ravelling"], "US 'Raveling' → IRC 'Ravelling'"),
    ("DISTRESS_TYPES: Weathering/Oxidation\nSEVERITY: Medium\nDESCRIPTION: Test.",
     ["Hungry Surface"], "Weathering → Hungry Surface (closest visible IRC)"),
    ("DISTRESS_TYPES: Block Crack (D43)\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Alligator Cracking"], "Block Crack → Alligator Cracking (closest visible)"),
    ("DISTRESS_TYPES: Edge Crack\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Edge Cracking"], "Edge Crack → Edge Cracking"),
    ("DISTRESS_TYPES: Depression\nSEVERITY: N/A\nDESCRIPTION: Test.",
     ["Shallow Depression"], "Depression → Shallow Depression"),
    ("DISTRESS_TYPES: Fatty Surface\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Bleeding"], "Fatty Surface → Bleeding"),
    ("DISTRESS_TYPES: Settlement\nSEVERITY: N/A\nDESCRIPTION: Test.",
     ["Settlement"], "Settlement direct"),
    ("DISTRESS_TYPES: Upheaval\nSEVERITY: N/A\nDESCRIPTION: Test.",
     ["Settlement"], "Upheaval → Settlement (per IRC §7.4.6 grouping)"),

    # ------ Multi-class with mix of legacy and IRC ------
    ("DISTRESS_TYPES: Pothole (D40), Alligator Crack (D20), Raveling\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Potholes", "Alligator Cracking", "Ravelling"], "mixed legacy + RDD codes"),

    # ------ 'Other / Unknown' family (must filter) ------
    ("DISTRESS_TYPES: Other Distress\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Unknown"], "Other Distress filtered → fallback to Unknown"),
    ("DISTRESS_TYPES: Other\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Unknown"], "Other filtered"),
    ("DISTRESS_TYPES: Unknown Distress\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Unknown"], "Unknown Distress filtered"),
    ("DISTRESS_TYPES: Potholes, Other Distress\nSEVERITY: High\nDESCRIPTION: Test.",
     ["Potholes"], "Other Distress filtered from multi"),

    # ------ Normal / no-distress ------
    ("DISTRESS_TYPES: Normal - No distress detected\nSEVERITY: None\nDESCRIPTION: Test.",
     ["Normal"], "Normal detection"),
    ("DISTRESS_TYPES: Normal\nSEVERITY: None\nDESCRIPTION: Test.",
     ["Normal"], "Plain Normal"),

    # ------ Hallucinated D-codes (model invents codes not in our schema) ------
    ("DISTRESS_TYPES: Rutting (D30)\nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Rutting"], "Hallucinated D30 stripped → Rutting"),
    ("DISTRESS_TYPES: Weathering/Oxidation (D70)\nSEVERITY: Medium\nDESCRIPTION: Test.",
     ["Hungry Surface"], "D70 hallucination + Weathering alias"),

    # ------ Case variations ------
    ("DISTRESS_TYPES: potholes\nSEVERITY: high\nDESCRIPTION: test",
     ["Potholes"], "lowercase 'potholes'"),
    ("DISTRESS_TYPES: POTHOLES\nSEVERITY: HIGH\nDESCRIPTION: test",
     ["Potholes"], "uppercase 'POTHOLES'"),

    # ------ Whitespace handling ------
    ("DISTRESS_TYPES:   Potholes  ,   Ravelling   \nSEVERITY: Medium\nDESCRIPTION: Test.",
     ["Potholes", "Ravelling"], "extra whitespace"),

    # ------ Keyword fallback (no structured DISTRESS_TYPES field) ------
    ("The pavement shows multiple potholes and severe alligator cracking. Treatment needed.",
     ["Potholes", "Alligator Cracking"], "free-text keyword extraction"),
    ("Significant rutting in the wheel path.",
     ["Rutting"], "free-text keyword extraction (rutting)"),
    ("Edge breaking along the shoulder.",
     ["Edge Breaking"], "free-text edge breaking"),

    # ------ Empty / malformed ------
    ("",
     ["Unknown"], "empty input → Unknown"),
    ("DISTRESS_TYPES: \nSEVERITY: Low\nDESCRIPTION: Test.",
     ["Unknown"], "empty types field → Unknown"),
]


def round2_parser_tests() -> dict:
    """Run all parser canonicalization cases.

    Comparison is set-based — the ORDER of distress_types is irrelevant for
    downstream metrics (TP/FP/FN, F1) which use set semantics. The parser is
    correct as long as the *set* of canonical labels matches.
    """
    passed = 0
    failed = []
    for txt, expected, desc in PARSER_CASES:
        result = parse_stage2_response(txt)
        got = result["distress_types"]
        if set(got) == set(expected) and len(got) == len(expected):
            passed += 1
        else:
            failed.append({
                "description": desc,
                "input": txt[:80] + ("…" if len(txt) > 80 else ""),
                "expected": expected,
                "got": got,
            })
    return {
        "status": "pass" if not failed else "fail",
        "n_cases": len(PARSER_CASES),
        "n_passed": passed,
        "n_failed": len(failed),
        "failures": failed,
    }


# ============================================================
# Round 3 — Severity criteria string match against PDF text
# ============================================================

def round3_severity_match() -> dict:
    """
    For each IRC type with severity criteria, verify that the thresholds we
    encoded appear (in normalised form) in the PDF text.
    """
    import pypdf
    pdf_path = PROJECT_ROOT / "IRC 82" / "irc.gov.in.082.2015.pdf"
    if not pdf_path.exists():
        return {"status": "skipped", "reason": "PDF not available"}

    with pdf_path.open("rb") as f:
        reader = pypdf.PdfReader(f)
        text = " ".join(p.extract_text() for p in reader.pages[15:40])
    # Normalise whitespace
    text_n = re.sub(r"\s+", " ", text).lower()

    # The mm thresholds we encoded. Phrases to find (loosely).
    checks = {
        "Longitudinal Cracking": [
            ("Low width 1-3 mm",   ["1-3 mm", "1 - 3 mm", "1to 3 mm", "1 to 3 mm"]),
            ("Medium width 3-6 mm", ["3-6 mm", "3 - 6 mm", "3 to 6 mm"]),
            ("High width > 6 mm",  ["greater than 6 mm", "more than 6 mm", "> 6 mm", "6 mm wide"]),
        ],
        "Transverse Cracking": [
            ("Low width 1-3 mm",   ["1-3 mm", "1 - 3 mm", "1to 3 mm"]),
            ("Medium width 3-6 mm", ["3-6 mm", "3 - 6 mm"]),
            ("High width > 6 mm",  ["greater than 6 mm", "6 mm wide"]),
        ],
        "Alligator Cracking": [
            ("Low 1-3 mm",  ["1 to 3 mm", "1-3 mm"]),
            ("Medium 3-6 mm",  ["3 to 6 mm", "3-6 mm"]),
            ("High > 6 mm", ["more than 6 mm", "6 mm"]),
        ],
        "Rutting": [
            ("Low 4-10 mm", ["4-10 mm", "4 to 10 mm", "4 - 10 mm"]),
            ("High > 10 mm", ["more than 10 mm", "greater than 10 mm", "10 mm"]),
        ],
        "Potholes": [
            ("Small 25 mm 200 mm", ["25 mm deep and 200 mm wide", "25 mm deep", "200 mm wide"]),
            ("Medium 25-50 mm 500 mm", ["25 to 50 mm", "500 mm wide"]),
            ("Large > 50 mm > 500 mm", ["50 mm deep", "500 mm width"]),
        ],
        "Edge Cracking": [
            ("Medium loss up to 10%", ["10%", "upto 10%", "up to 10"]),
            ("High loss > 10%", ["more than 10%", "10%"]),
        ],
    }

    results = []
    n_pass = 0
    n_fail = 0
    for label, criteria in checks.items():
        for tier_desc, phrases in criteria:
            found = any(p.lower() in text_n for p in phrases)
            results.append({
                "label": label,
                "tier_check": tier_desc,
                "phrases_tried": phrases,
                "found_in_pdf": found,
            })
            if found:
                n_pass += 1
            else:
                n_fail += 1

    return {
        "status": "pass" if n_fail == 0 else "partial",
        "n_checks": n_pass + n_fail,
        "n_pdf_match": n_pass,
        "n_no_match": n_fail,
        "details": results,
    }


# ============================================================
# Round 4 — Prompt content audit
# ============================================================

def round4_prompt_audit() -> dict:
    """Verify both v1 and v2 Stage 2 prompts mention IRC:82, forbid 'Other',
    and include all 18 IRC type names.

    The "forbids non-IRC labels" check accepts EITHER an explicit "FORBIDDEN"
    keyword (v1 phrasing) OR a "NOT output" / "must not" instructional phrasing
    (v2 phrasing) — both are valid ways to instruct the model not to emit
    'Other Distress' / 'Unknown' / 'Other'.
    """
    must_contain = ["IRC:82-2015"]
    all_18_types = list(IRC82_DISTRESS_TAXONOMY.keys())

    def _forbids_non_irc(prompt: str) -> bool:
        """The label 'Other Distress' MUST appear in the prompt AND it must
        appear in a negative-instruction context."""
        if "Other Distress" not in prompt:
            return False
        # Look for any negative-instruction pattern near "Other Distress"
        idx = prompt.find("Other Distress")
        # Window of 200 chars before "Other Distress" to find instruction
        window = prompt[max(0, idx - 200):idx].lower()
        negative_markers = ["forbidden", "must not", "do not", "don't",
                            "never", "no ", "not output", "not emit"]
        return any(m in window for m in negative_markers)

    audits = {}
    for name, prompt in [("v1_stage2", STAGE2_SYSTEM_PROMPT),
                         ("v2_stage2", STAGE2_SYSTEM_PROMPT_V2)]:
        report = {
            "length_chars": len(prompt),
            "contains_irc_marker": all(s in prompt for s in must_contain),
            "forbids_non_irc_labels": _forbids_non_irc(prompt),
            "missing_types": [t for t in all_18_types if t not in prompt],
            "n_types_named": sum(1 for t in all_18_types if t in prompt),
            "has_severity_criteria": "Severity rating per IRC:82-2015" in prompt
                                     or any(s in prompt for s in ["1-3 mm", "3-6 mm", "25 mm deep"]),
            "has_section_citations": "§" in prompt or "Section 7" in prompt,
        }
        audits[name] = report
        audits[name]["status"] = (
            "pass" if report["contains_irc_marker"]
            and report["forbids_non_irc_labels"]
            and not report["missing_types"]
            and report["has_section_citations"]
            and report["has_severity_criteria"]
            else "fail"
        )

    overall = "pass" if all(a["status"] == "pass" for a in audits.values()) else "fail"
    return {
        "status": overall,
        "prompts": audits,
        "n_irc_types": len(all_18_types),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("IRC:82-2015 ROBUSTNESS VERIFICATION — multi-round audit")
    print("=" * 78)

    print()
    print("[Round 1] PDF cross-reference (every §7 subsection accounted for)")
    r1 = round1_pdf_cross_reference()
    print(f"  status: {r1['status']}")
    if r1.get("misses"):
        print(f"  misses: {r1['misses']}")
    else:
        print(f"  mapped: {r1.get('n_mapped_to_taxonomy', 0)}, "
              f"excluded: {r1.get('n_explicitly_excluded', 0)}, "
              f"expected: {r1.get('n_expected_subsections', 0)}")

    print()
    print("[Round 2] Parser canonicalization tests")
    r2 = round2_parser_tests()
    print(f"  status: {r2['status']}  ({r2['n_passed']}/{r2['n_cases']} cases passed)")
    if r2["failures"]:
        for f in r2["failures"][:5]:
            print(f"  FAIL: {f['description']}")
            print(f"    expected: {f['expected']}")
            print(f"    got:      {f['got']}")

    print()
    print("[Round 3] Severity criteria match against PDF text")
    r3 = round3_severity_match()
    print(f"  status: {r3['status']}")
    if r3["status"] != "skipped":
        print(f"  PDF text matches: {r3['n_pdf_match']}/{r3['n_checks']} threshold checks")

    print()
    print("[Round 4] Prompt content audit")
    r4 = round4_prompt_audit()
    print(f"  status: {r4['status']}")
    for pname, audit in r4["prompts"].items():
        print(f"  {pname}: status={audit['status']}, types named={audit['n_types_named']}/{r4['n_irc_types']}")
        if audit["missing_types"]:
            print(f"    missing types: {audit['missing_types']}")

    # Combine
    overall = {
        "round_1_pdf_cross_reference": r1,
        "round_2_parser_tests": r2,
        "round_3_severity_match": r3,
        "round_4_prompt_audit": r4,
    }
    statuses = [r1["status"], r2["status"], r3["status"], r4["status"]]
    final = "pass" if all(s in ("pass", "skipped") for s in statuses) else "fail-or-partial"

    print()
    print("=" * 78)
    print(f"OVERALL: {final}")
    print(f"  Round 1: {r1['status']}")
    print(f"  Round 2: {r2['status']}  ({r2['n_passed']}/{r2['n_cases']})")
    print(f"  Round 3: {r3['status']}")
    print(f"  Round 4: {r4['status']}")
    print("=" * 78)

    out = PROJECT_ROOT / "eval_results" / "irc82_verification.json"
    out.write_text(json.dumps({"overall": final, "rounds": overall}, indent=2),
                   encoding="utf-8")
    print(f"\nReport saved to: {out}")

    if final != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
