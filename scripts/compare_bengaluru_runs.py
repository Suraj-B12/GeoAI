"""
Compare the 76 Bengaluru photos before and after reclassification with the
fine-tuned adapter + new system prompts.

Reads:
  eval_results/bengaluru_baseline_snapshot.json   (pre-reclass snapshot)
  Supabase assessments table                      (post-reclass current state)

Writes:
  eval_results/bengaluru_comparison.json
  eval_results/bengaluru_comparison_table.md      (paste-ready)

Match by id (UUID) since Supabase keeps the row, just overwrites the result fields.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = PROJECT_ROOT / "eval_results" / "bengaluru_baseline_snapshot.json"
SUPABASE_URL = "https://vtlkitpoffudiefuoijb.supabase.co"


def load_dotenv() -> str:
    """Read SUPABASE_SERVICE_KEY from .env."""
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        sys.exit("ERROR: .env not found")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "SUPABASE_SERVICE_KEY":
            return v.strip().strip('"').strip("'")
    sys.exit("ERROR: SUPABASE_SERVICE_KEY not in .env")


def fetch_current_rows(key: str) -> list[dict]:
    """Get all assessments that were re-processed (have processed_at)."""
    url = (
        f"{SUPABASE_URL}/rest/v1/assessments"
        "?processed_at=not.is.null&order=created_at.asc"
    )
    req = urllib.request.Request(url, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def compare(baseline: list[dict], current: list[dict]) -> dict:
    """Pair rows by id, compute per-row + aggregate deltas."""
    base_by_id = {r["id"]: r for r in baseline}
    curr_by_id = {r["id"]: r for r in current}

    all_ids = sorted(set(base_by_id) | set(curr_by_id))
    pairs = []
    for id_ in all_ids:
        b = base_by_id.get(id_)
        c = curr_by_id.get(id_)
        if b and c:
            pairs.append((b, c))

    # Aggregate counters
    status_xtab: dict[tuple[str, str], int] = Counter()
    s1_label_xtab: dict[tuple[str, str], int] = Counter()
    confidence_deltas_s1: list[float] = []
    confidence_deltas_s2: list[float] = []
    type_changes: list[dict] = []
    severity_changes: list[dict] = []
    review_flag_flips: dict[str, int] = Counter()

    for b, c in pairs:
        # Status crosstab
        status_xtab[(b["status"], c["status"])] += 1

        # Stage 1 label crosstab
        b_s1 = b.get("stage1_label") or "null"
        c_s1 = c.get("stage1_label") or "null"
        s1_label_xtab[(b_s1, c_s1)] += 1

        # Confidence deltas
        b_s1_conf = b.get("stage1_confidence") or 0
        c_s1_conf = c.get("stage1_confidence") or 0
        if b_s1_conf and c_s1_conf:
            confidence_deltas_s1.append(c_s1_conf - b_s1_conf)
        b_s2_conf = b.get("stage2_confidence") or 0
        c_s2_conf = c.get("stage2_confidence") or 0
        if b_s2_conf > 0 and c_s2_conf > 0:
            confidence_deltas_s2.append(c_s2_conf - b_s2_conf)

        # Type changes
        b_types = sorted(b.get("distress_types") or [])
        c_types = sorted(c.get("distress_types") or [])
        if b_types != c_types:
            type_changes.append({
                "id": b["id"][:8],
                "image_url": b.get("image_url"),
                "address": b.get("address"),
                "baseline_types": b_types,
                "finetuned_types": c_types,
            })

        # Severity change
        b_sev = b.get("severity") or "None"
        c_sev = c.get("severity") or "None"
        if b_sev != c_sev:
            severity_changes.append({
                "id": b["id"][:8],
                "baseline_severity": b_sev,
                "finetuned_severity": c_sev,
            })

        # Review flag flips
        b_flag = bool(b.get("needs_expert_review"))
        c_flag = bool(c.get("needs_expert_review"))
        if b_flag and not c_flag:
            review_flag_flips["resolved"] += 1
        elif not b_flag and c_flag:
            review_flag_flips["newly_flagged"] += 1
        else:
            review_flag_flips["unchanged"] += 1

    # Aggregate stats
    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "comparison_at": datetime.utcnow().isoformat() + "Z",
        "n_rows_compared": len(pairs),
        "n_baseline_only": len(base_by_id) - len(pairs),
        "n_current_only": len(curr_by_id) - len(pairs),
        "baseline_status_counts": dict(Counter(b["status"] for b in baseline)),
        "current_status_counts": dict(Counter(c["status"] for c in current)),
        "status_xtab": {f"{a} -> {b}": v for (a, b), v in status_xtab.items()},
        "stage1_label_xtab": {f"{a} -> {b}": v for (a, b), v in s1_label_xtab.items()},
        "review_flag_flips": dict(review_flag_flips),
        "confidence_delta_stage1_avg": _avg(confidence_deltas_s1),
        "confidence_delta_stage2_avg": _avg(confidence_deltas_s2),
        "n_type_changes": len(type_changes),
        "n_severity_changes": len(severity_changes),
    }
    return {
        "summary": summary,
        "type_changes": type_changes,
        "severity_changes": severity_changes,
    }


def to_markdown(result: dict) -> str:
    s = result["summary"]
    lines = []
    lines.append("# Bengaluru 76-Photo Re-classification Comparison")
    lines.append("")
    lines.append(f"Compared at: {s['comparison_at']}")
    lines.append(f"Rows compared: {s['n_rows_compared']}")
    lines.append("")
    lines.append("## Status distribution")
    lines.append("")
    lines.append("| Status | Baseline | Fine-tuned + new prompts |")
    lines.append("|---|---|---|")
    all_status = sorted(set(s["baseline_status_counts"]) | set(s["current_status_counts"]))
    for st in all_status:
        b = s["baseline_status_counts"].get(st, 0)
        c = s["current_status_counts"].get(st, 0)
        lines.append(f"| {st} | {b} | {c} |")
    lines.append("")
    lines.append("## Status transitions (baseline → fine-tuned)")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---|")
    for k, v in sorted(s["status_xtab"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Stage 1 label transitions")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---|")
    for k, v in sorted(s["stage1_label_xtab"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Expert-review flag flips")
    lines.append("")
    flips = s["review_flag_flips"]
    lines.append(f"- **Resolved** (was flagged, now isn't): {flips.get('resolved', 0)}")
    lines.append(f"- **Newly flagged** (wasn't flagged, now is): {flips.get('newly_flagged', 0)}")
    lines.append(f"- **Unchanged** (same flag value either way): {flips.get('unchanged', 0)}")
    lines.append("")
    lines.append("## Confidence deltas (avg, only over rows with both runs producing a value)")
    lines.append("")
    lines.append(f"- Stage 1 confidence delta avg: `{s['confidence_delta_stage1_avg']}`")
    lines.append(f"- Stage 2 confidence delta avg: `{s['confidence_delta_stage2_avg']}`")
    lines.append("")
    lines.append(f"## Type changes ({s['n_type_changes']} rows)")
    lines.append("")
    if result["type_changes"]:
        lines.append("| ID | Address | Baseline types | Fine-tuned types |")
        lines.append("|---|---|---|---|")
        for t in result["type_changes"][:30]:
            addr = (t.get("address") or "").replace("|", " ")[:40]
            lines.append(f"| `{t['id']}` | {addr} | {', '.join(t['baseline_types']) or '_(none)_'} | {', '.join(t['finetuned_types']) or '_(none)_'} |")
        if len(result["type_changes"]) > 30:
            lines.append(f"| ... | ... | _{len(result['type_changes']) - 30} more rows in JSON_ | |")
    else:
        lines.append("No type changes (every row has the same `distress_types` in both runs).")
    lines.append("")
    lines.append(f"## Severity changes ({s['n_severity_changes']} rows)")
    lines.append("")
    if result["severity_changes"]:
        lines.append("| ID | Baseline | Fine-tuned |")
        lines.append("|---|---|---|")
        for sv in result["severity_changes"][:20]:
            lines.append(f"| `{sv['id']}` | {sv['baseline_severity']} | {sv['finetuned_severity']} |")
    else:
        lines.append("No severity changes.")
    lines.append("")
    return "\n".join(lines)


def main():
    if not SNAPSHOT.exists():
        sys.exit(f"ERROR: snapshot not found at {SNAPSHOT}")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    baseline = snap["rows"]

    key = load_dotenv()
    print(f"Fetching current rows from Supabase...")
    current = fetch_current_rows(key)
    print(f"  {len(current)} rows currently processed")
    print(f"  {len(baseline)} rows in baseline snapshot")

    result = compare(baseline, current)

    # Save
    out_json = PROJECT_ROOT / "eval_results" / "bengaluru_comparison.json"
    out_md = PROJECT_ROOT / "eval_results" / "bengaluru_comparison_table.md"
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    out_md.write_text(to_markdown(result), encoding="utf-8")

    s = result["summary"]
    print()
    print("=" * 70)
    print("BENGALURU RE-CLASSIFICATION COMPARISON")
    print("=" * 70)
    print(f"Rows compared:         {s['n_rows_compared']}")
    print(f"Status — baseline:     {s['baseline_status_counts']}")
    print(f"Status — fine-tuned:   {s['current_status_counts']}")
    print(f"Type changes:          {s['n_type_changes']}")
    print(f"Severity changes:      {s['n_severity_changes']}")
    print(f"Review flag resolved:  {s['review_flag_flips'].get('resolved', 0)}")
    print(f"Newly flagged:         {s['review_flag_flips'].get('newly_flagged', 0)}")
    print(f"Stage 1 conf delta:    {s['confidence_delta_stage1_avg']}")
    print(f"Stage 2 conf delta:    {s['confidence_delta_stage2_avg']}")
    print()
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
